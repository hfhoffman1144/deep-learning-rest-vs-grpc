# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: inference.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0finference.proto\"5\n\x10InferenceRequest\x12\x12\n\nrequest_id\x18\x01 \x01(\x05\x12\r\n\x05texts\x18\x02 \x03(\t\"\x1b\n\tEmbedding\x12\x0e\n\x06values\x18\x01 \x03(\x02\"3\n\x11InferenceResponse\x12\x1e\n\nembeddings\x18\x01 \x03(\x0b\x32\n.Embedding2E\n\x0fInferenceServer\x12\x32\n\tinference\x12\x11.InferenceRequest\x1a\x12.InferenceResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'inference_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _globals['_INFERENCEREQUEST']._serialized_start=19
  _globals['_INFERENCEREQUEST']._serialized_end=72
  _globals['_EMBEDDING']._serialized_start=74
  _globals['_EMBEDDING']._serialized_end=101
  _globals['_INFERENCERESPONSE']._serialized_start=103
  _globals['_INFERENCERESPONSE']._serialized_end=154
  _globals['_INFERENCESERVER']._serialized_start=156
  _globals['_INFERENCESERVER']._serialized_end=225
# @@protoc_insertion_point(module_scope)