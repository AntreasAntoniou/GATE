import fire

from gate.config.assistant import assistance


class GateCLI(object):

    def config(self):
        assistance()


def main():
    fire.Fire(GateCLI)


if __name__ == "__main__":
    main()
