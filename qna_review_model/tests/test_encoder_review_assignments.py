from scripts.build_encoder_review_assignments import assign_products


def test_assign_products_keeps_product_together_and_balances():
    rows = []
    for index in range(9):
        for item in range(2):
            rows.append({
                "productGroupId": f"p{index}",
                "category": {"level1Name": "식품" if index % 2 else "생활/건강"},
                "itemId": f"r{index}-{item}",
            })
    result = assign_products(rows, reviewers=3, seed=7)
    assert len(result) == 9
    counts = [sum(value == reviewer for value in result.values()) for reviewer in range(3)]
    assert max(counts) - min(counts) <= 1
