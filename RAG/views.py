from django.shortcuts import render
from rest_framework import status, generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view
from RAG.serializers import EnterpriseDocumentSerializer,UserSerializer, ConversationSerializer, MessageSerializer, UserDocumentSerializer
from .models import EnterpriseDocuments,Users, Conversation, Message, UserDocuments
from .anonymization import anonymize, deAnonymize
from django.http import StreamingHttpResponse
# from .models import Conversation, Message
# from .serializers import ConversationSerializer, MessageSerializer

import os
import openai
# from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import AzureSearch
# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from django.core.files.storage import FileSystemStorage
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Create your views here.
@api_view(['POST'])
def RagResponse(request):
    query = request.data.get('query', None)
    conv_id = request.data.get('conv_id', None)

    if query is None:  
        return Response({'error': 'No query provided'}, status=status.HTTP_400_BAD_REQUEST)
    elif conv_id is None:
         return Response({'error': 'No conversation id provided'}, status=status.HTTP_400_BAD_REQUEST)
    # print(query)
    # print(conv_id)

    anonymized_prompt = anonymize(query)
    # print(anonymized_prompt)

    answer = process_query(anonymized_prompt, conv_id)  
    answer = deAnonymize(answer)

    userMsg = {
         'msg' : query,
         'conv_id' : conv_id,
         'msg_type' : "User"
    }
    # print(userMsg)
    userserializer = MessageSerializer(data=userMsg)
    print(userserializer)
    if userserializer.is_valid():
        userserializer.save()
    else:
         print('Not valid question')

    botMsg = {
        'msg' : answer,
        'conv_id' : conv_id,
        'msg_type' : "Bot"
    }

    botSerializer = MessageSerializer(data = botMsg)
    if botSerializer.is_valid():
        botSerializer.save()
        return Response(botSerializer.data, status=status.HTTP_201_CREATED)
    return Response(botSerializer.error_messages, status=status.HTTP_400_BAD_REQUEST)

    # Return the answer  
    return Response({'answer': answer}) 

def process_query(query, conv_id):
    OPENAI_API_BASE = "https://dwspoc.openai.azure.com/"
    OPENAI_API_KEY = "bd38ee31e244408cacab3e1dd4c32221"
    OPENAI_API_VERSION = "2024-02-15-preview"
    AZURE_COGNITIVE_SEARCH_SERVICE_NAME = 'enterprisegptaisearch'
    AZURE_COGNITIVE_SEARCH_API_KEY = "G2KtqsDYXA7dr0P5PSPLgxoTm01TJlcGZa4cBt3TdOAzSeCpDAXj"
    AZURE_COGNITIVE_SEARCH_INDEX_NAME = "e-gpt"
    vector_store_address= "https://enterprisegptaisearch.search.windows.net"
    vector_store_password= "G2KtqsDYXA7dr0P5PSPLgxoTm01TJlcGZa4cBt3TdOAzSeCpDAXj"

    openai.api_type = "azure"
    openai.base_url = OPENAI_API_BASE
    openai.api_key = OPENAI_API_KEY
    openai.api_version = OPENAI_API_VERSION

    conversations = Message.objects.filter(conv_id = conv_id)
    serializer = MessageSerializer(conversations, many=True)
    print(serializer.data)
    context=""
    # for obj in serializer.data[-6:]:  
    #     # get the msg_type and msg  
    #     msg_type = obj.get('msg_type')  
    #     msg = obj.get('msg')  
        
    #     # add the msg_type and msg to the string  
    #     context += f"{msg_type}: {msg}" 



    #initializing LLMs 

    llm = AzureChatOpenAI(deployment_name="GPT4", openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE, openai_api_version=OPENAI_API_VERSION,streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
    # embeddings = AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002", chunk_size=500, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE, openai_api_version=OPENAI_API_VERSION)
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2023-05-15",
        azure_endpoint=OPENAI_API_BASE,
        api_key= OPENAI_API_KEY)
    
    #connect to azure cognitive search
    acs = AzureSearch(azure_search_endpoint=vector_store_address, azure_search_key=vector_store_password, index_name=AZURE_COGNITIVE_SEARCH_INDEX_NAME, embedding_function=embeddings.embed_query)

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be standalone question. if the provided context does not have information, then say relevant information not found

    Chat history:
    {chat_history}
    Follow up input: {question}
    Standalone question:""")

    qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                                retriever=acs.as_retriever(),
                                                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                                                return_source_documents=True,
                                                verbose=False
                                                )
    
    chat_history = []
    question = 'Answer the following question using the provided context and chat history only, give least priority to chat history. If the provided context and chat history does not have any information about it then say "provided context does not have information about it". Chat history is'+context+' Query : '+query+'generate complete response in 10 seconds'
    result = qa({"question": question, "chat_history": chat_history})

    # print("Question:", query)
    # print("answer:", result["answer"])

    return result['answer']

@api_view(['GET'])
def getChatmsgs(request, conv_id):
    msgs = Message.objects.filter(conv_id=conv_id)
    serializer = MessageSerializer(msgs, many=True)
    return Response(serializer.data, status=status.HTTP_202_ACCEPTED)

@api_view(['POST'])
def updateMsg(request, id):
    try:
        message = Message.objects.get(id=id)

    except Message.DoesNotExist:
         return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'POST':
        serializer = MessageSerializer(message, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(status=status.HTTP_400_BAD_REQUEST)


    
@api_view(['POST'])
def DocumentUploadView(request):
    if 'document' not in request.FILES:  
            return Response({'error': 'No document in request.'}, status=status.HTTP_400_BAD_REQUEST) 
        
    document = request.FILES['document']  
    fs = FileSystemStorage(location= r'RAG\EnterpriseDocs\\')  
    filename = fs.save(document.name, document) 
    path = r'RAG\EnterpriseDocs\\' + filename
    docdata = {
         "edoc_path" : path
    }
   
    print(path)


    file_path = fs.path(filename) 
    print(file_path) 
    index_name = 'e-gpt'
    if processDocument(file_path,index_name) == 'Document added':
        serializer = EnterpriseDocumentSerializer(data = docdata)
        if serializer.is_valid():
             serializer.save()
        else:
             print('not a valid data')
        return Response({'message': 'Document uploaded and loaded successfully.', 'filename': filename}, status=status.HTTP_200_OK)
    
def processDocument(filepath, index_name):
        OPENAI_API_BASE = "https://dwspoc.openai.azure.com/"
        OPENAI_API_KEY = "bd38ee31e244408cacab3e1dd4c32221"
        OPENAI_API_VERSION = "2024-02-15-preview"
        AZURE_COGNITIVE_SEARCH_SERVICE_NAME = 'enterprisegptaisearch'
        AZURE_COGNITIVE_SEARCH_API_KEY = "G2KtqsDYXA7dr0P5PSPLgxoTm01TJlcGZa4cBt3TdOAzSeCpDAXj"
        AZURE_COGNITIVE_SEARCH_INDEX_NAME = index_name
        vector_store_address= "https://enterprisegptaisearch.search.windows.net"
        vector_store_password= "G2KtqsDYXA7dr0P5PSPLgxoTm01TJlcGZa4cBt3TdOAzSeCpDAXj"

        openai.api_type = "azure"
        openai.base_url = OPENAI_API_BASE
        openai.api_key = OPENAI_API_KEY
        openai.api_version = OPENAI_API_VERSION

        embeddings = AzureOpenAIEmbeddings(
                        azure_deployment="text-embedding-ada-002",
                        openai_api_version="2023-05-15",
                        azure_endpoint=OPENAI_API_BASE,
                        api_key= OPENAI_API_KEY
                    )
        
        #Connecting to azure cognitive search
        acs = AzureSearch(azure_search_endpoint=vector_store_address, azure_search_key=vector_store_password, index_name=AZURE_COGNITIVE_SEARCH_INDEX_NAME, embedding_function=embeddings.embed_query)
        
        loader = PyPDFLoader(filepath)
        document = loader.load()
        text_splitter = TokenTextSplitter(chunk_size=250, chunk_overlap=20)
        docs = text_splitter.split_documents(document)
        acs.add_documents(documents=docs)
        return 'Document added'


@api_view(['DELETE'])
def deleteDocument(request, id):
    try:
         document = EnterpriseDocuments.objects.get(pk=id) 
    except EnterpriseDocuments.DoesNotExist:
         return Response(status=status.HTTP_404_NOT_FOUND)
    serializer = EnterpriseDocumentSerializer(document)
    filepath = serializer.data['edoc_path']

    # Deleting file in filesystem

    print(filepath)
    if os.path.exists(filepath):  
        os.remove(filepath)  
        print("File deleted.")  
    else:  
        return Response(status=status.HTTP_404_NOT_FOUND)  
    
    #deleting the file path in database
    document.delete()

    #Updating the vector store
    endpoint = "https://enterprisegptaisearch.search.windows.net"
    admin_key = "G2KtqsDYXA7dr0P5PSPLgxoTm01TJlcGZa4cBt3TdOAzSeCpDAXj"  
    index_name = "e-gpt"
    credential = AzureKeyCredential(admin_key)  
    client = SearchIndexClient(endpoint, AzureKeyCredential(admin_key))
    client.delete_index(index_name)  

    pre_path = r'RAG\EnterpriseDocs\\'
    documents = []
    pdf_files = [f for f in os.listdir(pre_path) if f.endswith(".pdf")]  
    print(pdf_files)
    for i in pdf_files:
        processDocument(pre_path+i)
    
    return Response(status=status.HTTP_204_NO_CONTENT)
             
@api_view(['GET', 'POST'])             
def UserDetails(request):
     if request.method == 'GET':
        users = Users.objects.all()
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data)
     elif request.method == 'POST':
          serializer = UserSerializer(data=request.data)
          if serializer.is_valid():
               serializer.save()
               return Response(serializer.data, status=status.HTTP_201_CREATED)
          return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST) 
     
@api_view(['POST'])     
def ConvDetails(request):
    if request.method == 'POST':
        email_id = request.data.get('email_id', None)
        print(email_id)
        data = {
            'email_id' : email_id,
            'conv_name' : 'New Conversation'
        }
        serializer = ConversationSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            #conversation = Conversation.objects.get(email_id=email_id)
            return Response(serializer.data, status=status.HTTP_201_CREATED) 
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
@api_view(['GET'])
def getAllUserConv(request,email):
    if request.method == 'GET':
        conversations = Conversation.objects.filter(email_id = email)
        serializer = ConversationSerializer(conversations, many=True)
        return Response(serializer.data)
         

@api_view(['POST','PUT','GET','DELETE'])
def ConvDetailsPK(request, id):
    try:
        conversation = Conversation.objects.get(id=id)

    except Conversation.DoesNotExist:
         return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'POST':
        serializer = ConversationSerializer(conversation, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def userDocumetUpload(request):
    if 'document' not in request.FILES:  
            return Response({'error': 'No document in request.'}, status=status.HTTP_400_BAD_REQUEST)
    document = request.FILES['document']  
    email_id = request.data.get('email_id', None)
    fs = FileSystemStorage(location= r'RAG\UserDocs\\')  
    filename = fs.save(document.name, document) 
    path = r'RAG\UserDocs\\' + filename
    print(path)
    docdata = {
        'udoc_path' : path,
        'email_id' : email_id
    }

    print(docdata)

    file_path = fs.path(filename)
    # print(file_path)
    print(file_path)
    index_name = 'user-docs'
    if processDocument(file_path,index_name) == 'Document added':
        serializer = UserDocumentSerializer(data = docdata)
        if serializer.is_valid():
             serializer.save()
        else:
             print('not a valid data')
        return Response({'message': 'Document uploaded and loaded successfully.', 'filename': filename}, status=status.HTTP_200_OK)
   
@api_view(['DELETE'])   
def deleteUserDoc(request, id):
    try:
         document = UserDocuments.objects.get(pk=id) 
    except UserDocuments.DoesNotExist:
         return Response(status=status.HTTP_404_NOT_FOUND)
    
    serializer = UserDocumentSerializer(document)
    filepath = serializer.data['udoc_path']

    print(filepath)
    if os.path.exists(filepath):  
        os.remove(filepath)  
        print("File deleted.")  
    else:  
        return Response(status=status.HTTP_404_NOT_FOUND) 
    
    #deleting the file path in database
    document.delete()

    #Updating the vector store
    endpoint = "https://enterprisegptaisearch.search.windows.net"
    admin_key = "G2KtqsDYXA7dr0P5PSPLgxoTm01TJlcGZa4cBt3TdOAzSeCpDAXj"  
    index_name = 'user-docs'
    credential = AzureKeyCredential(admin_key)  
    client = SearchIndexClient(endpoint, AzureKeyCredential(admin_key))
    client.delete_index(index_name)
    
    pre_path = r'RAG\UserDocs\\'
    documents = []
    pdf_files = [f for f in os.listdir(pre_path) if f.endswith(".pdf")]  
    print(pdf_files)
    for i in pdf_files:
        processDocument(pre_path+i, index_name)

    return Response(status=status.HTTP_204_NO_CONTENT)
    


@api_view(['POST'])
def dummyApiCall(request):
    from langchain.chat_models import AzureChatOpenAI
    from langchain.schema import HumanMessage
    from openai import AzureOpenAI
    import json
    import os

    openai_api_base = "https://dwspoc.openai.azure.com/"
    openai_api_version = "2024-02-15-preview"
    deployment_name ="GPT4"
    openai_api_key = "bd38ee31e244408cacab3e1dd4c32221"
    openai_api_type="azure"
    client = AzureOpenAI(
                azure_endpoint = openai_api_base, 
                api_key=openai_api_key,  
                api_version=openai_api_version
            )
    query = request.data.get('query', None)
    annonyised_query = anonymize(query)
    response = client.chat.completions.create(
                model="GPT4", # model = "deployment_name".
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": annonyised_query},
                ]
            )
    
    
    print(annonyised_query)
    # answer = llm([HumanMessage(content=annonyised_query)])
    # answer = answer.to_dict()
    deAnonymised_answer = deAnonymize(response.choices[0].message.content)
    return Response({'answer' : deAnonymised_answer})


     
         
