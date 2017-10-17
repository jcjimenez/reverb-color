from argparse import ArgumentTypeError


class AtMost(object):
    def __init__(self, max_value):
        self._max_value = max_value

    def __call__(self, value):
        try:
            number = int(value)
        except ValueError:
            raise ArgumentTypeError('Not a number: {}'.format(value))

        if number > self._max_value:
            raise ArgumentTypeError('Value should be less or equal to {}'
                                    .format(self._max_value))

        return number
