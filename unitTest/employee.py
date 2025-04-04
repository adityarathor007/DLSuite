

class Employee: 

    raise_amt=1.05

    def __init__(self,first,last,pay):
        self.first=first
        self.last=last
        self.pay=pay

    
    #@property on a method, it becomes a getter. This means you can access the method like an attribute, but behind the scenes, it calls the method.
    @property
    def email(self):
        return '{}_{}@gmail.com'.format(self.first,self.last)
    
    @property
    def fullName(self):
        return '{} {}'.format(self.first,self.last)
    

    def apply_raise(self):
        self.pay= int(self.pay*self.raise_amt)


    def monthly_schedule(self,month):
        response=requests.get(f'http://company.com/{self.last}/{month}')
        if response.ok:
            return response.text
        else:
            return 'Bad Response!'