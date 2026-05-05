class AudioSpeechRecognition:
    @classmethod
    def transcribe(cls, model, paths2audio_files):
        transcription = model.transcribe(
            paths2audio_files=paths2audio_files, batch_size=1
        )
        return transcription

    @classmethod
    def transcribe_with_model(cls, model, paths2audio_files, decoder=None):
        logits = model.transcribe(paths2audio_files=paths2audio_files, logprobs=True)[0]
        if decoder:
            logits = decoder.decode(logits)
        return logits

    @classmethod
    def identify_language(cls, model, text):
        predictions = model.predict(text, k=1)
        lang_id = predictions[0][0]
        return lang_id

    @classmethod
    def identify_language_with_confidence(cls, model, text):
        predictions = model.predict(text, k=1)
        lang_id = predictions[0][0]
        confidence = predictions[1][0]
        return lang_id, confidence
