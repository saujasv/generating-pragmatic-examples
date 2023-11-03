from agents import ListenerOutput
from RegexPlus.synthesis import synthesize
from stopit import ThreadingTimeout as Timeout
from greenery.parse import NoMatch

class RegexPlusListener:
    def __init__(self):
        pass

    def synthesize(self, specs, return_scores=True):
        strings = [[s for s, l in spec if l == '+'] for spec in specs][0]
        if len(strings) == 0:
            return ListenerOutput(
                [[]],
                [[]],
                [[]],
                [[]]
                )
        with Timeout(6) as timeout_ctx:
            try:
                regexes = synthesize(strings)
            except NoMatch:
                regexes = []
        if timeout_ctx.state == timeout_ctx.EXECUTED:
            return ListenerOutput(
                [[str(r[1]) for r in regexes]],
                [[i for i, r in enumerate(regexes)]],
                [[str(r[1]) for r in regexes]],
                [[r[0] for r in regexes]]
                )
        else:
            return ListenerOutput(
                [[]],
                [[]],
                [[]],
                [[]]
                )

if __name__ == "__main__":
    listener = RegexPlusListener()
    print(listener.synthesize([("a", "+"), ("b", "+"), ("c", "+")]))