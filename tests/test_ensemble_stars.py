from pipeline.ensemble import compute_confidence_stars

def test_compute_confidence_stars():
    # 1 star: Base case (low edge, low prob)
    assert compute_confidence_stars(model_prob=0.50, edge=0.01) == 1
    
    # 2 stars: Prob >= 60%
    assert compute_confidence_stars(model_prob=0.65, edge=0.01) == 2
    
    # 2 stars: Edge >= 5%
    assert compute_confidence_stars(model_prob=0.50, edge=0.06) == 2
    
    # 3 stars: Prob >= 75%
    assert compute_confidence_stars(model_prob=0.80, edge=0.01) == 3
    
    # 3 stars: Edge >= 10%
    assert compute_confidence_stars(model_prob=0.50, edge=0.12) == 3
    
    # 3 stars: Prob >= 60% and Edge >= 5%
    assert compute_confidence_stars(model_prob=0.65, edge=0.06) == 3
    
    # 4 stars: Prob >= 75% and Edge >= 5%
    assert compute_confidence_stars(model_prob=0.80, edge=0.06) == 4
    
    # 4 stars: Prob >= 60% and Edge >= 10%
    assert compute_confidence_stars(model_prob=0.65, edge=0.12) == 4
    
    # 5 stars: Prob >= 75% and Edge >= 10%
    assert compute_confidence_stars(model_prob=0.80, edge=0.12) == 5
    
    # Capped at 5
    assert compute_confidence_stars(model_prob=0.99, edge=0.50) == 5
