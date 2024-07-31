import abc

class BaseTest(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def load_data(self):
        pass

    @abc.abstractmethod
    def load_models(self):
        pass

    @abc.abstractmethod
    def _pipline_setup(self):
        pass

    @abc.abstractmethod
    def run_test(self):
        pass 
    
    @abc.abstractmethod
    def report_ex(self):
        pass


