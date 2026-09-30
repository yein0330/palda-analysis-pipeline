from types import SimpleNamespace

import torch
from torch import nn

from scripts.train_encoder_transformer import PaldaEncoder, label_attention_pool


def test_attention_excludes_padding_and_supports_empty_mask():
    hidden = torch.tensor([[[1., 0.], [0., 2.], [900., 900.]]], requires_grad=True)
    queries = torch.tensor([[3., 0.], [0., 3.]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0]])
    context = label_attention_pool(hidden, mask, queries)
    assert context.shape == (1, 2, 2)
    assert context[0, 0, 0] > context[0, 1, 0]
    assert context[0, 1, 1] > context[0, 0, 1]
    context.sum().backward()
    assert torch.equal(hidden.grad[0, 2], torch.zeros(2))
    assert torch.isfinite(queries.grad).all() and queries.grad.abs().sum() > 0
    assert torch.equal(label_attention_pool(hidden, torch.zeros_like(mask), queries),
                       torch.zeros_like(context))


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(8, 4)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_model_backward_compatibility_and_attention_gradients(monkeypatch):
    monkeypatch.setattr('scripts.train_encoder_transformer.AutoModel.from_pretrained',
                        lambda *args, **kwargs: FakeBackbone())
    torch.manual_seed(3)
    original = PaldaEncoder('fake', 2).eval()
    torch.manual_seed(3)
    attention = PaldaEncoder('fake', 2, review_pooling='label-attention').eval()
    assert 'review_queries' not in original.state_dict()
    for name, value in original.state_dict().items():
        assert torch.equal(value, attention.state_dict()[name])
    PaldaEncoder('fake', 2).load_state_dict(original.state_dict(), strict=True)
    ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    old_output, new_output = original(ids, mask), attention(ids, mask)
    for key in ('primary', 'secondary', 'usable'):
        assert torch.equal(old_output[key], new_output[key])
    assert new_output['review'].shape == (2, 2, 3)
    new_output['review'].square().sum().backward()
    assert attention.review_queries.grad.abs().sum() > 0
    assert attention.review_head.weight.grad.abs().sum() > 0
    assert attention.backbone.embedding.weight.grad.abs().sum() > 0
    reloaded = PaldaEncoder('fake', 2, review_pooling='label-attention').eval()
    reloaded.load_state_dict(attention.state_dict(), strict=True)
    assert torch.equal(reloaded(ids, mask)['review'], new_output['review'])


def test_qna_mean_pooling_keeps_review_cls_and_state_compatibility(monkeypatch):
    monkeypatch.setattr('scripts.train_encoder_transformer.AutoModel.from_pretrained',
                        lambda *args, **kwargs: FakeBackbone())
    shared = PaldaEncoder('fake', 2).eval()
    split = PaldaEncoder('fake', 2, qna_pooling='mean').eval()
    mean = PaldaEncoder('fake', 2, pooling='mean').eval()
    split.load_state_dict(shared.state_dict(), strict=True)
    mean.load_state_dict(shared.state_dict(), strict=True)
    ids, mask = torch.tensor([[1, 2, 0]]), torch.tensor([[1, 1, 0]])
    shared_out, split_out, mean_out = shared(ids, mask), split(ids, mask), mean(ids, mask)
    assert torch.equal(shared_out['review'], split_out['review'])
    for key in ('primary', 'secondary', 'usable'):
        assert torch.equal(mean_out[key], split_out[key])
