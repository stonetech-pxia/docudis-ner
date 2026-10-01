// Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.

import 'dart:convert';
import 'dart:io';

import 'package:docudis_ner_ffi/docudis_ner_ffi.dart';
import 'package:test/test.dart';

String? _library() {
  final explicit = Platform.environment['DOCUDIS_NER_LIBRARY'];
  if (explicit != null) return explicit;
  for (final path in [
    '../../target/debug/libdocudis_ner_capi.dylib',
    '../../target/debug/libdocudis_ner_capi.so',
    r'..\..\target\debug\docudis_ner_capi.dll',
  ]) {
    if (File(path).existsSync()) return File(path).absolute.path;
  }
  return null;
}

const _skip = 'Build docudis-ner-capi first or set DOCUDIS_NER_LIBRARY';

void main() {
  final path = _library();
  // Real inference: DOCUDIS_NER_TEST_ORT names an ONNX Runtime library and
  // DOCUDIS_NER_TEST_MODEL a folder with model.json and its files.
  final runtime = Platform.environment['DOCUDIS_NER_TEST_ORT'];
  final modelDir = Platform.environment['DOCUDIS_NER_TEST_MODEL'];
  // The first ONNX Runtime load decides for the whole process, so a failed
  // one would make the real-inference test fail too.
  final failsRuntime = path == null
      ? _skip
      : runtime != null
      ? 'the real runtime is loaded instead'
      : false;

  test('the real library checks its ABI', () {
    expect(DocudisNerNative.open(path).abiVersion, 1);
  }, skip: path == null ? _skip : false);

  test('a missing runtime or model is a typed exception, not a crash', () {
    final native = DocudisNerNative.open(path);
    final spec = (jsonDecode(
      File('../../models/xlmr_ner_docudis/model.json').readAsStringSync(),
    ) as Map).cast<String, Object?>();
    expect(
      () => native.load(
        onnxRuntimeLibrary: '/definitely/missing/libonnxruntime.so',
        spec: spec,
        modelPath: '/missing/model.onnx',
        tokenizerPath: '/missing/tokenizer.json',
      ),
      throwsA(
        isA<DocudisNerException>()
            .having((e) => e.status, 'status', DocudisNerStatus.nerError)
            .having((e) => e.message, 'message', contains('ONNX Runtime')),
      ),
    );
    expect(
      () => native.load(
        onnxRuntimeLibrary: 'x',
        spec: {'name': 'broken'},
        modelPath: 'm',
        tokenizerPath: 't',
      ),
      throwsA(
        isA<DocudisNerException>().having(
          (e) => e.status,
          'status',
          DocudisNerStatus.invalidArgument,
        ),
      ),
    );
  }, skip: failsRuntime);

  test(
    'detects with UTF-16 offsets through the real library',
    () {
      final spec = (jsonDecode(
        File('$modelDir/model.json').readAsStringSync(),
      ) as Map).cast<String, Object?>();
      final model = DocudisNerNative.open(path).load(
        onnxRuntimeLibrary: runtime!,
        spec: spec,
        modelPath: '$modelDir/${spec['model']}',
        tokenizerPath: '$modelDir/${(spec['tokenizer']! as Map)['file']}',
      );
      const text =
          '😀 Bonjour, je suis Élodie Marchand, 12 rue des Lilas, Lyon.';
      final found = model.detect(text);
      final person = found.firstWhere((d) => d['type'] == 'PERSON');
      expect(
        text.substring(person['start']! as int, person['end']! as int),
        'Élodie Marchand',
      );
      expect(person['value'], 'Élodie Marchand');
      expect(person['source'], 'model');
      // The address reaches the same model, as another isolate would.
      expect(
        DocudisNerNative.open(path).model(model.address).detect(text),
        found,
      );
      model.close();
    },
    skip: path == null || runtime == null || modelDir == null
        ? 'set DOCUDIS_NER_TEST_ORT and DOCUDIS_NER_TEST_MODEL'
        : false,
  );
}
