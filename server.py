#!/usr/bin/env python3
"""Верстак Vosk, TeraTTSv2 и Edge TTS. Звук включает последний шаг."""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST, PORT = "127.0.0.1", 8765
LOCK = threading.Lock()
VOSK = None
TERA = None
JOBS = {}
VOSK_VOICES = {str(i) for i in range(5)}
TERA_VOICES = {"ru_f1", "ru_m5", "ru_f2", "ru_m1"}
EDGE_FALLBACK = [
    {"id": "ru-RU-SvetlanaNeural", "label": "Светлана, женский"},
    {"id": "ru-RU-DmitryNeural", "label": "Дмитрий, мужской"},
]
EDGE_CACHE = None
TERA_STAGES = [
    "Текст после ударений",
    "Индексы символов",
    "Стиль голоса",
    "Эмбеддинг текста",
    "Длительность",
    "Стартовый шум",
    "Латент сэмплера",
    "Волна вокодера",
]
VOSK_STAGES = [
    "Текст в синтезатор",
    "BERT по словам",
    "Фонемы",
    "Вход VITS",
    "Прогон сети",
    "Волна",
]
EDGE_STAGES = [
    "Запрос в Edge",
    "Ответ сервиса",
    "Волна",
]


def ensure_nix_libs():
    if os.environ.get("TTS_LIBS_READY") == "1":
        return
    gcc = subprocess.check_output(
        ["g++", "-print-file-name=libstdc++.so.6"], text=True
    ).strip()
    cache = ROOT / ".nix-libz-path"
    zlib = cache.read_text().strip() if cache.is_file() else ""
    if not (Path(zlib) / "lib/libz.so.1").is_file():
        zlib = subprocess.check_output(
            ["nix-build", "<nixpkgs>", "-A", "zlib", "--no-out-link"], text=True
        ).strip()
        cache.write_text(zlib + "\n")
    libs = [
        str(Path(gcc).parent),
        str(Path(zlib) / "lib"),
        "/run/opengl-driver/lib",
    ]
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvidia = Path(sys.prefix) / "lib" / py / "site-packages" / "nvidia"
    if nvidia.is_dir():
        libs.extend(str(path) for path in sorted(nvidia.glob("*/lib")) if path.is_dir())
    previous = os.environ.get("LD_LIBRARY_PATH")
    if previous:
        libs.append(previous)
    os.environ["LD_LIBRARY_PATH"] = ":".join(libs)
    os.environ["TTS_LIBS_READY"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def read_wav(write):
    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        write(handle.name)
        handle.seek(0)
        return handle.read()


def load_vosk():
    global VOSK
    if VOSK is None:
        from vosk_tts import Model, Synth
        VOSK = Synth(Model(model_name="vosk-model-tts-ru-0.9-multi"))
    return VOSK


def tera_device():
    import onnxruntime

    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        return "CUDAExecutionProvider", "CUDA"
    return "CPUExecutionProvider", "CPU"


def load_tera():
    global TERA
    if TERA is None:
        from transformers import AutoModel

        provider, accent = tera_device()
        TERA = AutoModel.from_pretrained(
            "TeraSpace/TeraTTSv2",
            trust_remote_code=True,
            provider=provider,
            ruaccent_device=accent,
            threads=4,
        )
        print(f"Tera: {provider}")
    return TERA


def np():
    import numpy
    return numpy


def stats(name, value):
    array = np().asarray(value)
    finite = array[np().isfinite(array)]
    if finite.size == 0:
        return f"{name}: shape={tuple(array.shape)} пусто"
    return (
        f"{name}: shape={tuple(array.shape)} dtype={array.dtype} "
        f"min={float(finite.min()):.4f} max={float(finite.max()):.4f} "
        f"mean={float(finite.mean()):.4f}"
    )


def grid(value, max_w=96, max_h=48):
    array = np().asarray(value, dtype=np().float32)
    array = np().squeeze(array)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim > 2:
        array = array.reshape(array.shape[-2], array.shape[-1])
    height, width = array.shape
    ys = np().linspace(0, height - 1, num=min(height, max_h), dtype=int)
    xs = np().linspace(0, width - 1, num=min(width, max_w), dtype=int)
    sample = array[np().ix_(ys, xs)]
    finite = sample[np().isfinite(sample)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 1.0
    if high == low:
        norm = np().zeros_like(sample)
    else:
        norm = (sample - low) / (high - low)
    return {
        "w": int(sample.shape[1]),
        "h": int(sample.shape[0]),
        "values": np().round(norm, 3).tolist(),
    }


def wave_bars(samples, bars=360):
    data = np().asarray(samples, dtype=np().float32).reshape(-1)
    if data.size == 0:
        return []
    if float(np().max(np().abs(data))) > 2:
        data = data / 32768.0
    edges = np().linspace(0, data.size, bars + 1, dtype=int)
    peaks = []
    for start, end in zip(edges[:-1], edges[1:]):
        chunk = data[start:end]
        peaks.append(round(float(np().max(np().abs(chunk))) if chunk.size else 0.0, 3))
    return peaks


def card(title, lines, plot=None, wave=None, audio=False, ms=0):
    return {
        "title": title,
        "lines": lines,
        "plot": plot,
        "wave": wave,
        "audio": audio,
        "ms": round(ms, 1),
    }


def remember(job):
    JOBS[job["id"]] = job
    while len(JOBS) > 4:
        JOBS.pop(next(iter(JOBS)))


def vosk_phonemes(synth, text):
    import re
    from vosk_tts.g2p import convert

    pattern = r"(\.\.\.|- |[ ,.?!;:\"()])"
    prepared = text.replace(" -", "- ")
    pieces = []
    for word in re.split(pattern, prepared.lower()):
        if word == "" or word == " ":
            continue
        if word in ("- ", "-") or re.match(pattern, word):
            pieces.append(word.strip())
            continue
        if word in synth.model.dic:
            pieces.append(synth.model.dic[word])
        else:
            pieces.append(convert(word))
    return " ".join(pieces)


def step_vosk(job):
    synth = load_vosk()
    bag = job["bag"]
    step = job["cursor"]
    started = time.perf_counter()
    text = job["text"].strip().replace("—", "-")
    if step == 0:
        bag["text"] = text
        return card("Текст в синтезатор", [
            text,
            f"голосов в модели: 0–4, выбран {job['voice']}",
            f"тип: {synth.model.config.get('model_type')}",
        ], ms=(time.perf_counter() - started) * 1000)
    if step == 1:
        bert = synth.get_word_bert(bag["text"], nopunc=True)
        bag["bert"] = bert
        array = np().asarray(bert)
        return card("BERT по словам", [stats("bert", array)], plot=grid(array), ms=(time.perf_counter() - started) * 1000)
    if step == 2:
        spoken = vosk_phonemes(synth, bag["text"])
        phoneme_ids, bert_embs = synth.g2p_multistream(bag["text"], bag["bert"])
        bag["phoneme_ids"] = phoneme_ids
        bag["bert_embs"] = bert_embs
        return card("Фонемы", [
            spoken,
            f"кадров после g2p: {len(phoneme_ids)}",
            "Словарь модели, если слово в нём есть. Иначе правило g2p.",
        ], ms=(time.perf_counter() - started) * 1000)
    if step == 3:
        ids = np().asarray(bag["phoneme_ids"], dtype=np().int64)
        embs = np().asarray(bag["bert_embs"], dtype=np().float32)
        bag["input"] = np().expand_dims(np().transpose(ids), 0)
        bag["bert_input"] = np().expand_dims(np().transpose(embs), 0)
        infer = synth.model.config["inference"]
        return card("Вход VITS", [
            stats("фонемы", bag["input"]),
            stats("bert", bag["bert_input"]),
            (
                f"noise={infer.get('noise_level', 0.8)} "
                f"rate={infer.get('speech_rate', 1.0)} "
                f"duration_noise={infer.get('duration_noise_level', 0.8)} "
                f"speaker={job['voice']}"
            ),
        ], ms=(time.perf_counter() - started) * 1000)
    if step == 4:
        infer = synth.model.config["inference"]
        scales = np().array([
            infer.get("noise_level", 0.8),
            1.0 / infer.get("speech_rate", 1.0),
            infer.get("duration_noise_level", 0.8),
        ], dtype=np().float32)
        args = {
            "input": bag["input"],
            "input_lengths": np().array([bag["input"].shape[2]], dtype=np().int64),
            "scales": scales,
            "sid": np().array([int(job["voice"])], dtype=np().int64),
            "bert": bag["bert_input"],
        }
        tick = time.perf_counter()
        audio = synth.model.onnx.run(None, args)[0].squeeze()
        spent = time.perf_counter() - tick
        audio = audio * infer.get("scale", 1.0)
        bag["audio"] = synth.audio_float_to_int16(audio)
        seconds = bag["audio"].shape[-1] / 22050
        return card("Прогон сети", [
            f"сеть считала {spent:.2f} с",
            f"звук {seconds:.2f} с",
            f"реальное время: {spent / seconds:.2f}" if seconds else "пусто",
            "частота 22050 Гц, дальше ещё не пишется wav",
        ], ms=(time.perf_counter() - started) * 1000)
    peaks = wave_bars(bag["audio"])
    job["wav"] = pcm16_wav(bag["audio"], 22050)
    return card("Волна", [
        f"отсчётов: {bag['audio'].shape[-1]}",
        "Звук молчит, пока не нажмёшь «Запустить звук».",
    ], wave=peaks, audio=True, ms=(time.perf_counter() - started) * 1000)


def pcm16_wav(samples, rate):
    import io
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np().asarray(samples, dtype=np().int16).tobytes())
    return buffer.getvalue()


def bounded(value, low, high, default):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return min(high, max(low, number))


def tera_controls(payload, voice):
    duration = bounded(payload.get("duration", 1.15), 0.75, 1.35, 1.15)
    blend = bounded(payload.get("blend", 0), 0.0, 1.0, 0.0)
    other = str(payload.get("blend_voice", ""))
    if other not in TERA_VOICES or other == voice:
        other = ""
        blend = 0.0
    seed = int(bounded(payload.get("seed", 1234), 0, 9999, 1234))
    return duration, other, blend, seed


def tera_style(runtime, voice):
    voice_dir = runtime.release / "styles" / voice
    ttl = np().load(voice_dir / "style_ttl.npy").astype(np().float32, copy=True)
    dp = np().load(voice_dir / "style_dp.npy").astype(np().float32, copy=True)
    return ttl, dp


def mix_styles(base_ttl, base_dp, other_ttl, other_dp, amount):
    ttl = (1 - amount) * base_ttl + amount * other_ttl
    dp = (1 - amount) * base_dp + amount * other_dp
    return ttl.astype(np().float32), dp.astype(np().float32)


def step_tera(job):
    model = load_tera()
    runtime = model.runtime
    bag = job["bag"]
    step = job["cursor"]
    started = time.perf_counter()
    tagged = f"<ru>{job['text']}</ru>"
    if step == 0:
        bag["model_text"] = model.normalize_text(tagged)
        provider, accent = tera_device()
        return card("Текст после ударений", [
            bag["model_text"],
            f"устройство сети: {provider}, ударения: {accent}",
            "Плюс перед гласной — ударение, которое поставил RUAccent.",
        ], ms=(time.perf_counter() - started) * 1000)
    if step == 1:
        bag["text_ids"], bag["text_mask"] = runtime.indexer.batch(bag["model_text"])
        bare = bag["model_text"].replace("+", "")
        bag["duration_ids"], bag["duration_mask"] = runtime.indexer.batch(bare)
        pairs = []
        for char, token in zip(bag["model_text"][:240], bag["text_ids"][0].tolist()):
            shown = "␠" if char == " " else char
            pairs.append(f"{shown}:{token}")
        return card("Индексы символов", [
            " ".join(pairs),
            stats("text_ids", bag["text_ids"]),
            f"без плюсов ударения, для предиктора длины: {len(bare)} символов",
        ], ms=(time.perf_counter() - started) * 1000)
    if step == 2:
        ttl, dp = tera_style(runtime, job["voice"])
        other = job.get("blend_voice") or ""
        amount = float(job.get("blend") or 0)
        if other and amount:
            other_ttl, other_dp = tera_style(runtime, other)
            ttl, dp = mix_styles(ttl, dp, other_ttl, other_dp, amount)
            mix_line = f"примешано {other} на {amount:.2f}"
        else:
            mix_line = "чужой стиль не примешан"
        bag["style_ttl"] = ttl
        bag["style_dp"] = dp
        return card("Стиль голоса", [
            stats("style_ttl", bag["style_ttl"]),
            stats("style_dp", bag["style_dp"]),
            "Два вектора стиля: один в текстовый энкодер, другой в длительность.",
            mix_line,
        ], plot=grid(bag["style_ttl"]), ms=(time.perf_counter() - started) * 1000)
    if step == 3:
        bag["text_emb"] = runtime.text_encoder.run(None, {
            "text_ids": bag["text_ids"],
            "style_ttl": bag["style_ttl"],
            "text_mask": bag["text_mask"],
        })[0]
        return card("Эмбеддинг текста", [
            stats("text_emb", bag["text_emb"]),
        ], plot=grid(bag["text_emb"]), ms=(time.perf_counter() - started) * 1000)
    if step == 4:
        raw = runtime.duration_predictor.run(None, {
            "text_ids": bag["duration_ids"],
            "style_dp": bag["style_dp"],
            "text_mask": bag["duration_mask"],
        })[0]
        raw_value = float(np().asarray(raw).reshape(-1)[0])
        scale = float(job.get("duration") or 1)
        bag["duration_seconds"] = raw_value / 1.05 * scale
        bag["latent_length"] = max(1, int(np().ceil(bag["duration_seconds"] * 44100 / 3072)))
        return card("Длительность", [
            f"сырое предсказание: {raw_value:.4f}",
            "делится на 1.05, так устроена эта модель",
            f"скорость {scale:.2f}: больше — речь длиннее и медленнее",
            f"секунды: {bag['duration_seconds']:.3f}",
            f"кадров латента: {bag['latent_length']} (кадр = 3072 отсчёта при 44100 Гц)",
        ], ms=(time.perf_counter() - started) * 1000)
    if step == 5:
        seed = int(job.get("seed", 1234))
        bag["noise"] = np().random.default_rng(seed).standard_normal(
            (1, 144, bag["latent_length"])
        ).astype(np().float32)
        return card("Стартовый шум", [
            stats("noise", bag["noise"]),
            f"Зерно {seed}. Другое число — другой розыгрыш, не большая громкость шума.",
            "Сэмплер потом переписывает этот тензор.",
        ], plot=grid(bag["noise"]), ms=(time.perf_counter() - started) * 1000)
    if step == 6:
        mask = np().ones((1, 1, bag["latent_length"]), dtype=np().float32)
        bag["latent"] = runtime.sampler.run(None, {
            "initial_latent": bag["noise"],
            "text_emb": bag["text_emb"],
            "style_ttl": bag["style_ttl"],
            "latent_mask": mask,
            "text_mask": bag["text_mask"],
            "guidance": np().asarray([3.0], dtype=np().float32),
        })[0]
        return card("Латент сэмплера", [
            stats("latent", bag["latent"]),
            "Это ещё не звук. Вокодер читает этот тензор.",
        ], plot=grid(bag["latent"]), ms=(time.perf_counter() - started) * 1000)
    waveform = runtime.vocoder.run(None, {"latent": bag["latent"]})[0]
    maximum = round(bag["duration_seconds"] * 44100)
    bag["wave"] = np().asarray(waveform[0, :maximum], dtype=np().float32)
    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        model.save_wav(handle.name, bag["wave"])
        handle.seek(0)
        job["wav"] = handle.read()
    return card("Волна вокодера", [
        stats("wave", bag["wave"]),
        f"обрезано до {maximum} отсчётов по предсказанной длительности",
        "Звук молчит, пока не нажмёшь «Запустить звук».",
    ], wave=wave_bars(bag["wave"]), audio=True, ms=(time.perf_counter() - started) * 1000)


def edge_voice_catalog():
    global EDGE_CACHE
    if EDGE_CACHE is not None:
        return EDGE_CACHE
    try:
        import asyncio
        import edge_tts

        raw = asyncio.run(edge_tts.list_voices())
    except Exception:
        return list(EDGE_FALLBACK)
    found = []
    for voice in raw:
        if voice.get("Locale") != "ru-RU":
            continue
        short = voice.get("ShortName") or ""
        if not short:
            continue
        gender = {"Female": "женский", "Male": "мужской"}.get(voice.get("Gender"), "")
        name = short.split(":", 1)[0].split("-")[-1]
        if name.endswith("Neural"):
            name = name[:-6]
        name = {
            "Svetlana": "Светлана",
            "Dmitry": "Дмитрий",
            "Dariya": "Дарья",
        }.get(name, name)
        found.append({"id": short, "label": f"{name}, {gender}" if gender else name})
    found.sort(key=lambda item: (0 if item["label"].endswith("женский") else 1, item["label"]))
    EDGE_CACHE = found or list(EDGE_FALLBACK)
    return EDGE_CACHE


def edge_prosody(payload):
    rate = int(bounded(payload.get("rate", 0), -50, 100, 0))
    volume = int(bounded(payload.get("volume", 0), -50, 100, 0))
    pitch = int(bounded(payload.get("pitch", 0), -100, 100, 0))
    return f"{rate:+d}%", f"{volume:+d}%", f"{pitch:+d}Hz"


def fetch_edge_mp3(text, voice, rate, volume, pitch):
    import asyncio
    import edge_tts

    async def run():
        communicate = edge_tts.Communicate(
            text, voice, rate=rate, volume=volume, pitch=pitch
        )
        chunks = []
        async for message in communicate.stream():
            if message["type"] == "audio":
                chunks.append(message["data"])
        return b"".join(chunks)

    data = asyncio.run(run())
    if not data:
        raise RuntimeError("Edge вернул пустой звук")
    return data


def decode_mp3(data):
    import miniaudio

    sound = miniaudio.decode(data, nchannels=1, sample_rate=24000)
    return np().asarray(sound.samples, dtype=np().int16), int(sound.sample_rate)


def step_edge(job):
    bag = job["bag"]
    step = job["cursor"]
    started = time.perf_counter()
    if step == 0:
        return card("Запрос в Edge", [
            job["text"],
            f"голос: {job['voice']}",
            "Текст уходит на сервер Microsoft. Своей модели на диске нет.",
            f"скорость {job['rate']}, высота {job['pitch']}, громкость {job['volume']}",
        ], ms=(time.perf_counter() - started) * 1000)
    if step == 1:
        tick = time.perf_counter()
        bag["mp3"] = fetch_edge_mp3(
            job["text"], job["voice"], job["rate"], job["volume"], job["pitch"]
        )
        spent = time.perf_counter() - tick
        return card("Ответ сервиса", [
            f"mp3, {len(bag['mp3'])} байт",
            f"сервис отвечал {spent:.2f} с",
            "Дальше файл только разбирается в волну.",
        ], ms=(time.perf_counter() - started) * 1000)
    samples, rate = decode_mp3(bag["mp3"])
    bag["audio"] = samples
    job["wav"] = pcm16_wav(samples, rate)
    seconds = samples.shape[-1] / rate if rate else 0
    return card("Волна", [
        f"отсчётов: {samples.shape[-1]}",
        f"частота {rate} Гц, звук {seconds:.2f} с",
    ], wave=wave_bars(samples), audio=True, ms=(time.perf_counter() - started) * 1000)


def stages_for(engine):
    if engine == "tera":
        return TERA_STAGES
    if engine == "edge":
        return EDGE_STAGES
    return VOSK_STAGES


def run_step(job):
    stages = stages_for(job["engine"])
    if job["cursor"] >= len(stages):
        raise ValueError("цепочка уже досчитана")
    if job["engine"] == "tera":
        result = step_tera(job)
    elif job["engine"] == "edge":
        result = step_edge(job)
    else:
        result = step_vosk(job)
    result["step"] = job["cursor"]
    result["total"] = len(stages)
    job["cursor"] += 1
    return result


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            body = (ROOT / "index.html").read_bytes()
            self.respond(200, "text/html; charset=utf-8", body)
            return
        if path == "/voices":
            body = json.dumps({"edge": edge_voice_catalog()}, ensure_ascii=False).encode()
            self.respond(200, "application/json; charset=utf-8", body)
            return
        if path.startswith("/bench/") and path.endswith(".wav"):
            job = JOBS.get(path[len("/bench/"):-len(".wav")])
            if job is None or not job.get("wav"):
                self.send_error(404)
                return
            self.respond(200, "audio/wav", job["wav"])
            return
        self.send_error(404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            if path == "/bench":
                result = self.new_bench(payload)
            elif path.startswith("/bench/") and path.endswith("/step"):
                result = self.next_step(path[len("/bench/"):-len("/step")], payload)
            else:
                self.send_error(404)
                return
        except Exception as exc:
            self.respond(400, "text/plain; charset=utf-8", str(exc).encode())
            return
        self.respond(200, "application/json; charset=utf-8", json.dumps(result, ensure_ascii=False).encode())

    def new_bench(self, payload):
        text = str(payload.get("text", "")).strip()
        engine = payload.get("engine")
        voice = str(payload.get("voice", ""))
        if not text:
            raise ValueError("пустой текст")
        if len(text) > 2000:
            raise ValueError("текст длиннее 2000 знаков")
        duration, blend_voice, blend, seed = 1.0, "", 0.0, 1234
        rate, volume, pitch = "+0%", "+0%", "+0Hz"
        if engine == "vosk" and voice in VOSK_VOICES:
            stages = VOSK_STAGES
        elif engine == "tera" and voice in TERA_VOICES:
            stages = TERA_STAGES
            duration, blend_voice, blend, seed = tera_controls(payload, voice)
        elif engine == "edge" and any(item["id"] == voice for item in edge_voice_catalog()):
            stages = EDGE_STAGES
            rate, volume, pitch = edge_prosody(payload)
        else:
            raise ValueError("неизвестный движок или голос")
        job = {
            "id": uuid.uuid4().hex,
            "engine": engine,
            "voice": voice,
            "text": text,
            "duration": duration,
            "blend_voice": blend_voice,
            "blend": blend,
            "seed": seed,
            "rate": rate,
            "volume": volume,
            "pitch": pitch,
            "cursor": 0,
            "bag": {},
            "wav": None,
        }
        with LOCK:
            remember(job)
        return {"id": job["id"], "stages": stages}

    def next_step(self, job_id, payload):
        with LOCK:
            job = JOBS.get(job_id)
            if job is None:
                raise ValueError("цепочка устарела, собери её заново")
            wanted = payload.get("step")
            if wanted is not None and int(wanted) != job["cursor"]:
                raise ValueError("этот шаг уже пройден или ещё рано")
            return run_step(job)

    def respond(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(fmt % args)


if __name__ == "__main__":
    ensure_nix_libs()
    if "--check" in sys.argv:
        import numpy
        import onnxruntime
        import torch
        print("ok", numpy.__version__, torch.__version__, onnxruntime.__version__)
        raise SystemExit
    print(f"http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
