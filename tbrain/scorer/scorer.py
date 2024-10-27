from tbrain.scorer.precision_scorer import PrecisionScorer


class Scorer:
    @classmethod
    def get_scorer(cls, name: str):
        if name == "precision":
            return PrecisionScorer
        else:
            raise ValueError(f"Unsupported scorer: {name}")
