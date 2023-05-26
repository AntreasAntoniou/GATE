from typing import Any


class MultiFCTask:
    def __init__(self):
        super().__init__()
        self.labels = {}

    def __call__(self, inputs) -> Any:
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            Any: _description_

        .. code-block:: python
        An input looks like this before processing
        {
        'claimID': 'pomt-14885',
        'claim': 'Says the United States is in "the worst recovery from an economic recession since World War II."',
        'label': 'half-true',
        'claimURL': '/truth-o-meter/statements/2015/nov/10/chris-christie/christie-us-worst-recovery-wwii/',
        'reason': 'The central theme of the Republican presidential debate in Milwaukee, Wis., was business and economics. Fox Business modera',
        'categories': None,
        'speaker': 'Chris Christie',
        'checker': None,
        'tags': None,
        'article title': None,
        'publish date': '2015-11-10T21:50:05',
        'climate': '2015-11-10',
        'entities': "['United_States']"
        }

        Please note that the train dataset has 27.9K rows with labels distributed as follows:
        {
            'unsubstantiated messages', 'truth!', '0', 'misleading recommendations', 'fact', 'determination: barely true', 'fake news',
            'no evidence', 'fake', 'confirmed authorship!', 'in the works', 'stalled', 'a little baloney', 'in-between', 'correct attribution!',
            'grass roots movement!', 'incorrect', 'fiction', '2', 'verdict: unsubstantiated', 'truth! & misleading!', 'half-true', 'determination: mostly true',
            'determination: false', 'investigation pending!', 'determination: a stretch', 'mostly-correct', 'mostly_true', 'in-the-red', 'factscan score: misleading',
            'disputed!', 'verdict: false', 'verified', 'miscaptioned', '3', 'promise kept', 'some baloney', 'determination: huckster propaganda', 'rating: false',
            'unsupported', 'factscan score: true', 'unobservable', 'outdated!', 'mixture', 'misleading!', 'true', 'spins the facts', 'full flop', 'distorts the facts',
            'mostly fiction!', 'we rate this claim false', 'authorship confirmed!', 'mostly truth!', '3 pinnochios', 'conclusion: accurate', 'scam!', 'incorrect
            attribution!', 'needs context', 'a lot of baloney', 'not yet rated', 'mostly_false', 'virus!', 'truth! & outdated!', 'false', 'truth! & unproven!',
            'mostly false', '1', 'in-the-green', 'accurate', 'determination: misleading', 'inaccurate attribution!', 'cherry picks', None, '4', '10', 'promise
            broken', 'factscan score: false', '4 pinnochios', 'legend', 'bogus warning', 'correct attribution', 'none', 'opinion!', '2 pinnochios', 'fiction!',
            'statirical reports', 'partly true', 'determination: true', 'truth! & fiction!', 'correct', 'conclusion: unclear', 'not the whole story', 'half flip',
            'understated', 'exaggerates', 'unproven', 'verdict: true', 'misattributed', 'commentary!', 'outdated', 'half true', 'facebook scams', 'no flip', 'pants
            on fire!', 'scam', 'unproven!', 'previously truth! now resolved!', 'partially true', 'fiction! & satire!', 'truth! & disputed!', 'true messages',
            'conclusion: false', 'compromise', 'exaggerated', 'unverified', 'mostly true', 'misleading'
        }
        We took no decision in unifiying these labels under a common index and left this to the user chosen transformation.
        """
        return {
            "text": {
                "premise": inputs["claim"],
                "hypothesis": inputs["reason"],
                "metadata": {
                    "claimID": inputs["claimID"],
                    "claimURL": inputs["claimURL"],
                    "categories": inputs["categories"],
                    "speaker": inputs["speaker"],
                    "checker": inputs["checker"],
                    "tags": inputs["tags"],
                    "article_title": inputs["article title"],
                    "publish_date": inputs["publish date"],
                    "climate": inputs["climate"],
                    "entities": inputs["entities"],
                },
            },
            "labels": inputs["label"],
        }
