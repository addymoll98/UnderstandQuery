class Understand_File:
    
    def __init__(self, relname, depends, dependsby, contents, functions=[]):
        self.relname = relname
        self.depends = depends
        self.dependsby = dependsby
        self.contents = contents
        self.functions = functions
    
    def append_func(self, function):
        self.functions.append(function)

class Understand_Function:

    def __init__(self, fullname, content, ref_list):
        self.fullname = fullname
        self.content = content
        self.ref_list = ref_list