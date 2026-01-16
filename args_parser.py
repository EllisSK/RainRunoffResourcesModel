from argparse import ArgumentParser

class CustomParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--rainfall", action="store_true", help="Utilise rainfall modelling")
        self.add_argument("--climate", action="store_true", help="Utilise temperature and pet modelling")
        self.add_argument("--shetran", action="store_true", help="Utilise shetran discharge modelling")
        self.add_argument("--resource", action="store_true", help="Utilise pywr water resource modelling")
