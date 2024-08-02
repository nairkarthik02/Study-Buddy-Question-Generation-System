import argparse
import logging
import os
import uuid

from vosk import Model, SetLogLevel

from audio_transcriber import Transcribe


def main():
    print("parser engaged")
    parser = argparse.ArgumentParser()
    print("parser defined")
    parser.add_argument("-v", "--video", type=str, required=False, default="")
    parser.add_argument("-a", "--audio", type=str, required=False, default="")
    args = parser.parse_args()

    print("arguments passed")

    if not (args.video or args.audio):
        parser.error("No action requested, add --video or --audio")
    elif args.video and args.audio:
        parser.error("Only select one action --video or --audio")

    SetLogLevel(-1)
    print("model loading pre step")
    # model = Model(model_path="vosk-model-en-us-0.42-gigaspeech")
    model = Model(model_path="vosk-model-en-in-0.5")
    print("Model loaded")
    logging.info("sp2t setup")

    print("creating unique id for new video file")

    video_id = str(uuid.uuid4())

    print("making directory for a video file")

    os.makedirs(f"{video_id}")

    if args.audio != "":
        audio = args.audio
        print("audio file loaded directly")
    else:
        audio = f"{video_id}/audio.wav"
        print("audio file extracted from video file")

    Sharetape = Transcribe(
        args.video,
        audio,
        f"{video_id}/mono_audio.wav",
        f"{video_id}/transcript.txt",
        f"{video_id}/words.json",
        f"{video_id}/captions.srt",
        model,
    )
    Sharetape.extract_transcript()


if __name__ == "__main__":
    main()
