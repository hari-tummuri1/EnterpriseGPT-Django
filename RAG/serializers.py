from rest_framework import serializers  
from .models import Users, Conversation, Message, UserDocuments, EnterpriseDocuments
  
# class MessageSerializer(serializers.ModelSerializer):  
#     class Meta:  
#         model = Message  
#         fields = ['user', 'text', 'timestamp']  
  
# class ConversationSerializer(serializers.ModelSerializer):  
#     messages = MessageSerializer(many=True, read_only=True)  
  
#     class Meta:  
#         model = Conversation  
#         fields = ['id', 'user', 'start_time', 'end_time', 'messages'] 

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = Users
        fields = ['email_id', 'isAdmin'] 

class ConversationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conversation
        fields = ['id','email_id', 'conv_name','pinned']

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id','msg', 'conv_id', 'msg_type', 'feedback','time']

class UserDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserDocuments
        fields = ['udoc_path', 'email_id']

class EnterpriseDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = EnterpriseDocuments
        fields = ['edoc_path']


