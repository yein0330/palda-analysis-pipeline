from types import SimpleNamespace

import torch

from scripts.train_encoder_transformer import batch_loss


def test_review_task_weight_only_scales_review_direct_gradient():
    outputs={'review':torch.zeros(2,2,3,requires_grad=True),
             'primary':torch.zeros(2,2,requires_grad=True),
             'secondary':torch.zeros(2,2,requires_grad=True),
             'usable':torch.zeros(2,requires_grad=True)}
    batch={'review_task':torch.tensor([True,False]),'sample_weights':torch.ones(2),
           'non_special_mask':torch.ones(2,2,dtype=torch.bool),'category_mask':torch.ones(2,2,dtype=torch.bool),
           'review_targets':torch.zeros(2,2,3),'primary_targets':torch.tensor([-100,0]),
           'secondary_targets':torch.zeros(2,2),'usable_targets':torch.ones(2)}
    args=SimpleNamespace(review_pos_weight=3.,review_loss_weight=1.,secondary_pos_weight=3.,
                         secondary_loss_weight=.5,usable_loss_weight=.2)
    base,parts=batch_loss(outputs,batch,args)
    base_grad=torch.autograd.grad(base,(outputs['review'],outputs['primary']),retain_graph=True)
    args.review_loss_weight=3.
    scaled,_=batch_loss(outputs,batch,args)
    gradients=torch.autograd.grad(scaled,(outputs['review'],outputs['primary']))
    assert torch.allclose(gradients[0],3*base_grad[0])
    assert torch.allclose(gradients[1],base_grad[1])
    assert abs(float(scaled-base)-2*parts['review'])<1e-6
