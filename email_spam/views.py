from django.shortcuts import render

# Create your views here.
from rest_framework import permissions, serializers, status
from rest_framework.generics import GenericAPIView

# Create your views here.
from rest_framework.response import Response
import client

class TrainModelSerializer(serializers.Serializer):
    max_features = serializers.IntegerField()

class EmailClassifySerializer(serializers.Serializer):
    email_string = serializers.CharField()



class TrainModelView(GenericAPIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = TrainModelSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        max_features = serializer.validated_data.get('max_features')
        response = client.train_model(max_features)
        return Response(response, status=status.HTTP_200_OK)


class email_classify(GenericAPIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = EmailClassifySerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        email_string = serializer.validated_data.get('email_string')
        response = client.email_classify(email_string)
        return Response(response, status=status.HTTP_200_OK)