import pytest
import torch

from scripts.train_encoder_transformer import pool_tokens


def test_mean_pooling_excludes_padding_and_preserves_cls():
    hidden=torch.tensor([[[2.,4.],[4.,8.],[100.,100.]]])
    mask=torch.tensor([[1,1,0]])
    assert torch.equal(pool_tokens(hidden,mask,'mean'),torch.tensor([[3.,6.]]))
    assert torch.equal(pool_tokens(hidden,mask,'cls'),hidden[:,0])
    assert torch.isfinite(pool_tokens(hidden,torch.zeros_like(mask),'mean')).all()
    with pytest.raises(ValueError):
        pool_tokens(hidden,mask,'unknown')
