// Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'offsets.dart';

final class _Buffer extends Struct {
  external Pointer<Uint8> ptr;
  @Size()
  external int len;
  @Size()
  external int capacity;
}

final class _Model extends Opaque {}

typedef _VersionNative = Uint32 Function();
typedef _VersionDart = int Function();
typedef _LoadNative = Int32 Function(
  Pointer<Uint8>,
  Size,
  Pointer<Pointer<_Model>>,
);
typedef _LoadDart = int Function(Pointer<Uint8>, int, Pointer<Pointer<_Model>>);
typedef _DetectNative = Int32 Function(
  Pointer<_Model>,
  Pointer<Uint8>,
  Size,
  Pointer<_Buffer>,
);
typedef _DetectDart = int Function(
  Pointer<_Model>,
  Pointer<Uint8>,
  int,
  Pointer<_Buffer>,
);
typedef _ModelFreeNative = Void Function(Pointer<_Model>);
typedef _ModelFreeDart = void Function(Pointer<_Model>);
typedef _BufferFreeNative = Void Function(Pointer<_Buffer>);
typedef _BufferFreeDart = void Function(Pointer<_Buffer>);
typedef _ErrorNative = Pointer<Char> Function();
typedef _ErrorDart = Pointer<Char> Function();

enum DocudisNerStatus {
  ok,
  invalidArgument,
  invalidUtf8,
  invalidJson,
  nerError,
  panic,
  unknown,
}

class DocudisNerException implements Exception {
  const DocudisNerException(this.status, this.message);
  final DocudisNerStatus status;
  final String message;
  @override
  String toString() => 'DocudisNerException(${status.name}): $message';
}

/// The Docudis NER native library (`libdocudis_ner_capi`).
///
/// Opening it is cheap and may be repeated in any isolate; a loaded model is
/// identified by its [DocudisNerModel.address], which can cross isolates.
class DocudisNerNative {
  DocudisNerNative._(DynamicLibrary library)
    : _abiVersion = library.lookupFunction<_VersionNative, _VersionDart>(
        'docudis_ner_v1_abi_version',
      ),
      _load = library.lookupFunction<_LoadNative, _LoadDart>(
        'docudis_ner_v1_load_json',
      ),
      _detect = library.lookupFunction<_DetectNative, _DetectDart>(
        'docudis_ner_v1_detect_json',
      ),
      _modelFree = library.lookupFunction<_ModelFreeNative, _ModelFreeDart>(
        'docudis_ner_v1_model_free',
      ),
      _bufferFree = library.lookupFunction<_BufferFreeNative, _BufferFreeDart>(
        'docudis_ner_v1_buffer_free',
      ),
      _lastError = library.lookupFunction<_ErrorNative, _ErrorDart>(
        'docudis_ner_v1_last_error_message',
      ) {
    final actual = _abiVersion();
    if (actual != expectedAbiVersion) {
      throw DocudisNerException(
        DocudisNerStatus.invalidArgument,
        'incompatible Docudis NER ABI $actual; expected $expectedAbiVersion',
      );
    }
  }

  factory DocudisNerNative.open([String? path]) =>
      DocudisNerNative._(DynamicLibrary.open(path ?? defaultLibraryName));

  static String get defaultLibraryName {
    if (Platform.isMacOS || Platform.isIOS) return 'libdocudis_ner_capi.dylib';
    if (Platform.isWindows) return 'docudis_ner_capi.dll';
    return 'libdocudis_ner_capi.so';
  }

  static const expectedAbiVersion = 1;

  final _VersionDart _abiVersion;
  final _LoadDart _load;
  final _DetectDart _detect;
  final _ModelFreeDart _modelFree;
  final _BufferFreeDart _bufferFree;
  final _ErrorDart _lastError;

  int get abiVersion => _abiVersion();

  /// Loads ONNX Runtime (once per process) and the model described by
  /// [spec], the decoded `model.json`. [onnxRuntimeLibrary] is a path or a
  /// bare file name the platform loader resolves, such as the
  /// `libonnxruntime.so` an Android app ships.
  DocudisNerModel load({
    required String onnxRuntimeLibrary,
    required Map<String, Object?> spec,
    required String modelPath,
    required String tokenizerPath,
  }) {
    final input = _encode({
      'schema_version': 1,
      'onnxruntime_library': onnxRuntimeLibrary,
      'spec': spec,
      'model_path': modelPath,
      'tokenizer_path': tokenizerPath,
    });
    final out = calloc<Pointer<_Model>>();
    try {
      _check(_load(input.$1, input.$2, out));
      return DocudisNerModel._(this, out.value);
    } finally {
      calloc.free(out);
      calloc.free(input.$1);
    }
  }

  /// The model loaded elsewhere (for instance in another isolate) at
  /// [address].
  DocudisNerModel model(int address) =>
      DocudisNerModel._(this, Pointer<_Model>.fromAddress(address));

  (Pointer<Uint8>, int) _encode(Map<String, Object?> request) {
    final bytes = utf8.encode(jsonEncode(request));
    final input = calloc<Uint8>(bytes.length);
    input.asTypedList(bytes.length).setAll(0, bytes);
    return (input, bytes.length);
  }

  void _check(int status) {
    if (status == 0) return;
    final message = _lastError().cast<Utf8>().toDartString();
    throw DocudisNerException(_status(status), message);
  }

  static DocudisNerStatus _status(int code) => switch (code) {
    0 => DocudisNerStatus.ok,
    1 => DocudisNerStatus.invalidArgument,
    2 => DocudisNerStatus.invalidUtf8,
    3 => DocudisNerStatus.invalidJson,
    4 => DocudisNerStatus.nerError,
    255 => DocudisNerStatus.panic,
    _ => DocudisNerStatus.unknown,
  };
}

class DocudisNerModel {
  DocudisNerModel._(this._native, this._pointer);

  final DocudisNerNative _native;
  final Pointer<_Model> _pointer;

  /// Identifies the model across isolates; see [DocudisNerNative.model].
  int get address => _pointer.address;

  /// Detections in [text] as schema-v1 JSON objects with half-open UTF-16
  /// offsets, the indexing of Dart strings.
  List<Map<String, Object?>> detect(String text) {
    final input = _native._encode({'schema_version': 1, 'text': text});
    final output = calloc<_Buffer>();
    try {
      _native._check(_native._detect(_pointer, input.$1, input.$2, output));
      // Copy before handing the allocation back to Rust.
      final bytes = Uint8List.fromList(
        output.ref.ptr.asTypedList(output.ref.len),
      );
      final response = jsonDecode(utf8.decode(bytes)) as Map<String, Object?>;
      final offsets = Utf8ToUtf16Offsets(text);
      return [
        for (final raw in response['detections']! as List<Object?>)
          {
            ...(raw! as Map).cast<String, Object?>(),
            'start': offsets[(raw as Map)['start']! as int],
            'end': offsets[raw['end']! as int],
          },
      ];
    } finally {
      if (output.ref.ptr.address != 0) _native._bufferFree(output);
      calloc.free(output);
      calloc.free(input.$1);
    }
  }

  /// Releases the model. It must not be used afterwards, in any isolate.
  void close() => _native._modelFree(_pointer);
}
