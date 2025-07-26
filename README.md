# BBB transcription bot

![Project Logo](img/transcription-pipeline.png)

## Overview

This project is part of my bachelor's thesis and serves as a pipeline for processing the audio stream from [BigBlueButton (BBB)](https://bigbluebutton.org/). It utilizes a [BBB bot](https://github.com/bigbluebutton-bot/bigbluebutton-bot) to add live subtitles to BBB meetings by converting live speech into text in real-time. It uses the [stream_pipeline](https://github.com/bigbluebutton-bot/stream_pipeline)-framework to process the audio stream.

**🚧Note:** This project is no longer actively developed here. For ongoing development, please visit the [BBB Translation Bot repository](https://github.com/bigbluebutton-bot/bbb-translation-bot).

## Features

- **Live Transcription:** Converts live speech in BBB to text using advanced speech recognition models.
- **Real-Time Subtitles:** Adds the transcribed text as subtitles in BBB meetings.
- **Simulation Mode:** Allows testing the transcription pipeline with pre-recorded audio files.
- **Modular Architecture:** Each module handles a specific task within the pipeline, ensuring scalability and maintainability.

## Getting Started

Follow these simple steps to quickly set up the BBB-Translation-Bot.

### Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/bigbluebutton-bot/bbb-transcription-bot.git
   cd bbb-transcription-bot
   ```

2. **Configure Environment Variables:**
    If you only want to run the simulation, you can skip this step. And go to [Simulation](#simulation)

   - Duplicate the `.env_example` file and rename it to `.env`.

     ```bash
     cp .env_example .env
     ```

    - Open the `.env` file and configure the necessary environment variables as per your setup. In particular set `STT_WS_URL` to the WebSocket endpoint of your speech-to-text engine.

3. **Start the Application:**

   ```bash
   docker compose up
   ```

   This command will build and start all necessary services defined in the `docker-compose.yml` file.

## Usage

Once the application is running, it will crreate a meeting and process the audio stream from this BBB meeting and provide live subtitles. This version was only tested with BBB 2.5.

## Simulation

A simulation mode is available to test the transcription pipeline without needing a live BBB session.

### Steps to Use Simulation:

1. **Add an Audio File:**

   - Place your audio file (any format) into the `audio` folder of the project.

2. **Run the Simulation:**

   ```bash
   docker compose -f docker-compose-simulation.yml up
   ```

   This command will start a simulation that plays the audio file as if it were live. The simulation results will be stored in the `simulate_results` folder.

3. **Review Results:**

   - The `simulate_results` folder contains graphs and output files that display the Word Error Rate (WER) and latency metrics, indicating how long it takes for spoken words to be transcribed.

**Note:** Multiple simulations may run one after the other, which can extend the total processing time.

## Modules

The transcription pipeline is built using a modular architecture, where each module handles a specific task to efficiently process the audio stream from BBB and generate accurate live transcripts. This design ensures scalability, maintainability, and ease of testing.

### Key Modules

1. **Audio Buffer Module**

   - **Purpose:** Creates a n-second audio buffer from the live audio stream to ensure optimal performance with the STT service.
   - **Functionality:** Implements an OGG reader to parse audio data into manageable chunks, maintaining a continuous buffer without introducing significant delays. This module is stateful and operates in a non-parallel mode to preserve buffer integrity.

2. **Flow Limiter Module**
   
   - **Purpose:** Controls the rate at which audio data packets enter the pipeline to manage processing load.
   - **Functionality:** Limits the number of data packets processed per second, preventing overload and ensuring consistent subtitle delivery. Operates in a non-parallel mode to maintain controlled data flow.

3. **Convert Audio Module**
   
   - **Purpose:** Standardizes incoming audio data into a uniform format suitable for processing.
   - **Functionality:** Utilizes `ffmpeg` to convert various audio formats to 16-bit linear PCM (mono, 16 kHz), then transforms the binary data into normalized NumPy arrays. This stateless module supports parallel processing with controlled data sequencing.

4. **VAD (Voice Activity Detection) Module**
   
   - **Purpose:** Identifies and isolates segments of the audio stream that contain speech.
   - **Functionality:** Detects speech activity to filter out silent passages, forwarding only relevant audio segments to the transcription model. Uses the same VAD model as WhisperX for enhanced accuracy and operates in a stateless, parallel mode.

5. **WebSocket STT Module**

   - **Purpose:** Sends the buffered audio to an external speech-to-text service over WebSocket.
   - **Functionality:** Streams the audio buffer to the STT engine defined by `STT_WS_URL` and converts the returned JSON into timestamped text segments. This module replaces the built-in Whisper model when configured.

6. **Confirm Words Module**
   
   - **Purpose:** Ensures the stability and accuracy of transcribed words over time.
   - **Functionality:** Confirms words after they have been stable for a configurable period (e.g., 2 seconds), preventing further modifications and maintaining transcript integrity. Handles exceptions for low-probability words and overlapping timestamps, ensuring consistent and reliable transcription results. This stateful module operates in a non-parallel mode to manage confirmed word states effectively.

Each module plays a critical role in the live transcription process, working together to deliver real-time, accurate subtitles in BBB meetings. The modular approach allows for easy updates and maintenance, leveraging the strengths of each component to build a robust transcription pipeline.

## License

This project is licensed under the [MIT License](LICENSE).

---

For further development and updates, please visit the [BBB Translation Bot repository](https://github.com/bigbluebutton-bot/bbb-translation-bot).