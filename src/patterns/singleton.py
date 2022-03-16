class SingletonInstance:
    __instance = None

    @classmethod
    def __getInstance(cls):
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kwargs):
        cls.__instance = cls(*args, **kwargs)
        cls.instace = cls.__getInstance
        return cls.__instance
