from django.db import models
# from django.contrib.auth.models import User

# Create your models here.
# class Conversation(models.Model):  
#     user = models.ForeignKey(User, on_delete=models.CASCADE)  
#     start_time = models.DateTimeField(auto_now_add=True)  
#     end_time = models.DateTimeField(null=True, blank=True)

# class Message(models.Model):  
#     user = models.ForeignKey(User, on_delete=models.CASCADE)  
#     conversation = models.ForeignKey(Conversation, related_name='messages', on_delete=models.CASCADE)  
#     text = models.TextField()  
#     timestamp = models.DateTimeField(auto_now_add=True)

class Users(models.Model):
    email_id = models.CharField(max_length=50)
    # name = models.CharField(max_length=10)
    # password = models.CharField(max_length=20)
    isAdmin =  models.BooleanField(default=False)

    def __str__(self):
        return self.email_id+self.isAdmin

class Conversation(models.Model):
    email_id = models.CharField(max_length=50)
    # conv_id = models.IntegerField(default=1, unique=True)
    conv_name = models.CharField(max_length=20)
    pinned = models.BooleanField(default=False)

    def __str__(self):
        return self.email_id+self.conv_name

class Message(models.Model):
    # msg_id = models.IntegerField(default=1, unique=True)
    msg = models.TextField()
    conv_id = models.IntegerField() 
    msg_type = models.CharField(max_length=20)
    feedback = models.CharField(max_length=10, default='neutral')
    time = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.msg+self.conv_id+self.msg_type+self.time

class UserDocuments(models.Model):
    # udoc_id = models.IntegerField(default=1, unique=True)
    udoc_path = models.CharField(max_length=100)
    email_id =  models.CharField(max_length=50)

class EnterpriseDocuments(models.Model):
    # edoc_id = models.IntegerField(default=1, unique=True)
    edoc_path = models.CharField(max_length=100)
