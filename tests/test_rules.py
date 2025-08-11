from twinuav.rules import RuleEngine

def test_rule_once():
    r = RuleEngine({'obj': (0,0,0)})
    assert r.once('hit', r.near('obj', (0.1,0.1,0.1), 0.5)) is True
    assert r.once('hit', True) is False