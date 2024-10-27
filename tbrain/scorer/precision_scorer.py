import json


class PrecisionScorer:
    def __init__(self, answer_path: str):
        with open(answer_path, "r", encoding="utf8") as f:
            ground_truths_dict = json.load(f)

        self.ground_truths = {}
        for ground_truth in ground_truths_dict["ground_truths"]:
            self.ground_truths[ground_truth["qid"]] = ground_truth["retrieve"]

    def score(self, output_path: str):
        with open(output_path, "r", encoding="utf8") as f:
            answer_dict = json.load(f)

        assert len(answer_dict["answers"]) == len(self.ground_truths)

        correct = 0
        for answer in answer_dict["answers"]:
            if answer["retrieve"] == self.ground_truths[answer["qid"]]:
                correct += 1
        precision = correct / len(answer_dict["answers"])
        print(f"Precision: {round(precision, 7)}")
