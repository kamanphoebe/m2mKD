# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        # outputs_kd = None
        # if not isinstance(outputs, torch.Tensor):
        #     # assume that the model outputs a tuple of [outputs, outputs_kd]
        #     outputs, outputs_kd = outputs
        outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


class m2mKDLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, device, base_criterion, meta, teacher, student, student_name,
                 idx, codes, states, distillation_type, alpha_teacher, tau):
        assert distillation_type in ['none', 'soft', 'hard']
        super().__init__()
        self.device = device
        self.base_criterion = base_criterion
        self.meta = meta
        self.teacher = teacher
        self.student = student
        self.student_name = student_name
        self.codes = codes
        self.states = states
        self.idx = idx
        self.distillation_type = distillation_type
        self.alpha_teacher = alpha_teacher
        self.alpha_student = 1 - alpha_teacher
        self.tau = tau

    def meta_forward(self, inputs):
        if self.idx == 0:
            return None, None
        end = sum(self.meta.div[0: self.idx])
        meta_output = self.meta(inputs, start=0, end=end)
        return meta_output

    def get_teacher_outputs(self, inputs, meta_output):
        (idx, start, end, teacher) = self.teacher
        if idx == 0:
            teacher_outputs = teacher(inputs, start, end)
        else:
            teacher_outputs = teacher(meta_output, start, end)
            if self.idx == -1:
                return teacher_outputs
        teacher_outputs = self.meta(teacher_outputs, start=idx+1, end=len(self.meta.stages))
        return teacher_outputs
    
    def get_nac_outputs(self, inputs, meta_output):
        model = self.student
        if self.idx == 0:
            nac_outputs = model(inputs, self.states, self.codes)
        elif self.idx == -1:
            nac_outputs = model(self.states, self.codes, meta_output, self.codes)
            return nac_outputs
        else:
            nac_outputs = model(meta_output, self.codes, meta_output, self.codes)
        nac_outputs = self.meta(nac_outputs, self.idx+1, len(self.meta.stages))
        return nac_outputs
    
    def get_vmoe_outputs(self, inputs, meta_output):
        model = self.student
        if self.idx == 0:
            vmoe_outputs = model(inputs)
        else:
            vmoe_outputs = model(meta_output)
            if self.idx == -1:
                return vmoe_outputs
        vmoe_outputs = self.meta(vmoe_outputs, sum(self.meta.div[:self.idx+1]), sum(self.meta.div))
        return vmoe_outputs

    def forward(self, inputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            labels: the labels for the base criterion
        """
        # Don't backprop through the teacher and meta model.
        with torch.no_grad():
            meta_output = self.meta_forward(inputs)
            teacher_outputs = self.get_teacher_outputs(inputs, meta_output)
        if 'vmoe' in self.student_name:
            student_outputs = self.get_vmoe_outputs(inputs, meta_output)
        else:
            student_outputs = self.get_nac_outputs(inputs, meta_output)
        base_loss = self.base_criterion(student_outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if self.distillation_type == 'soft':
            T = self.tau
            teacher_loss = F.kl_div(
                F.log_softmax(student_outputs / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / student_outputs.numel()
        elif self.distillation_type == 'hard':
            teacher_loss = F.cross_entropy(student_outputs, teacher_outputs.argmax(dim=1))

        loss = base_loss * self.alpha_student + teacher_loss * self.alpha_teacher
        return loss
