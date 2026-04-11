"""Download voice-cloning profiles from the HuggingFace dataset repo."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import urllib.request
from urllib.error import HTTPError

log = logging.getLogger(__name__)

CUSTOM_PROFILES_REPO_URL = os.environ.get("MAZINGER_PROFILES_REPO_URL")
DEFAULT_PROFILES_REPO_URL = "https://huggingface.co/datasets/bakrianoo/mazinger-dubber-profiles/resolve/main/profiles"
PROFILES_REPO_URL = DEFAULT_PROFILES_REPO_URL
HF_TOKEN = os.environ.get("HF_TOKEN")

SCRIPT_FILENAME = "script.txt"
VOICE_EXTENSIONS = ("wav", "m4a", "mp3")

MIN_CLONE_DURATION = 20.0
MAX_CLONE_DURATION = 60.0
MIN_CLONE_WORDS = 20


def _make_request(url: str, method: str = "GET") -> urllib.request.Request:
    req = urllib.request.Request(url, method=method)
    if HF_TOKEN:
        req.add_header("Authorization", f"Bearer {HF_TOKEN}")
    return req


def _download_file(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    log.info("Downloading %s -> %s", url, dest)
    with urllib.request.urlopen(_make_request(url), timeout=30) as response:
        with open(dest, "wb") as f:
            f.write(response.read())


def _convert_to_wav(src: str, dest: str) -> None:
    """Convert any audio file to 16-kHz mono WAV using ffmpeg."""
    log.info("Converting %s -> %s", src, dest)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", src,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            dest,
        ],
        check=True,
        capture_output=True,
    )


def _download_voice(base_url: str, profile_dir: str) -> str:
    """Try each supported extension and return the path to the downloaded file."""
    for ext in VOICE_EXTENSIONS:
        dest = os.path.join(profile_dir, f"voice.{ext}")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            log.info("Using cached voice: %s", dest)
            return dest
        try:
            _download_file(f"{base_url}/voice.{ext}", dest)
            return dest
        except HTTPError:
            # Remove any empty/partial file left by the failed download
            if os.path.exists(dest):
                os.remove(dest)
            continue
    raise FileNotFoundError(
        f"No voice file found for profile at {base_url} "
        f"(tried extensions: {', '.join(VOICE_EXTENSIONS)})"
    )


def _ensure_wav(voice_path: str) -> str:
    """Return a WAV version of *voice_path*, converting if necessary."""
    if voice_path.endswith(".wav"):
        return voice_path
    wav_path = os.path.splitext(voice_path)[0] + ".wav"
    if os.path.exists(wav_path):
        return wav_path
    _convert_to_wav(voice_path, wav_path)
    return wav_path



# ═══════════════════════════════════════════════════════════════════════════════
#  Voice Themes (Qwen VoiceDesign)
# ═══════════════════════════════════════════════════════════════════════════════

_THEME_REF_TEXTS = {
    "Chinese": (
        "大家好，今天我们将一起探索一些非常精彩的内容。"
        "从古老的丝绸之路到现代的量子物理学，"
        "每一个发现都充满了惊喜和挑战。"
        "请仔细聆听，因为这些知识将彻底改变你对世界的认知，"
        "让我们共同踏上这段奇妙的旅程吧。"
    ),
    "English": (
        "Welcome everyone, today we are going to explore something truly fascinating. "
        "Throughout history, philosophers and scientists have questioned the very fabric of existence, "
        "pushing boundaries through observation, experimentation, and sheer curiosity. "
        "From the depths of the ocean to the vast stretches of outer space, "
        "each breakthrough challenge reshapes our understanding of the world around us."
    ),
    "Japanese": (
        "皆さんこんにちは、今日はとても魅力的なテーマを一緒に探っていきましょう。"
        "歴史を通じて、哲学者や科学者たちは存在の本質を問い続けてきました。"
        "深海の神秘から宇宙の果てまで、"
        "観察と実験を重ねるたびに新しい発見が生まれ、"
        "私たちの世界観は大きく変わってきたのです。"
    ),
    "Korean": (
        "여러분 안녕하세요, 오늘은 정말 흥미로운 주제를 함께 살펴보겠습니다. "
        "역사를 통해 철학자와 과학자들은 존재의 본질에 대해 끊임없이 질문해 왔습니다. "
        "깊은 바다의 신비부터 광활한 우주의 끝자락까지, "
        "관찰과 실험을 거듭할수록 새로운 발견이 탄생하며 "
        "우리의 세계관은 크게 변화해 왔습니다."
    ),
    "German": (
        "Herzlich willkommen, heute werden wir gemeinsam etwas wirklich Faszinierendes erkunden. "
        "Im Laufe der Geschichte haben Philosophen und Wissenschaftler die Grundfesten unserer Existenz hinterfragt, "
        "Grenzen durch Beobachtung, Experimente und pure Neugier verschoben. "
        "Von den Tiefen der Ozeane bis zu den Weiten des Weltraums "
        "verändert jeder Durchbruch unser Verständnis der Welt um uns herum."
    ),
    "French": (
        "Bienvenue à tous, aujourd'hui nous allons explorer ensemble quelque chose de vraiment fascinant. "
        "À travers l'histoire, philosophes et scientifiques ont questionné le tissu même de l'existence, "
        "repoussant les frontières grâce à l'observation, l'expérimentation et la pure curiosité. "
        "Des profondeurs de l'océan aux vastes étendues de l'espace, "
        "chaque percée transforme notre compréhension du monde qui nous entoure."
    ),
    "Russian": (
        "Добро пожаловать, сегодня мы вместе исследуем нечто поистине увлекательное. "
        "На протяжении всей истории философы и учёные ставили под вопрос саму природу бытия, "
        "раздвигая границы через наблюдение, эксперимент и чистое любопытство. "
        "От глубин океана до бескрайних просторов космоса "
        "каждое открытие меняет наше представление об окружающем мире."
    ),
    "Portuguese": (
        "Bem-vindos, hoje vamos explorar juntos algo verdadeiramente fascinante. "
        "Ao longo da história, filósofos e cientistas questionaram a própria essência da existência, "
        "expandindo fronteiras através da observação, experimentação e pura curiosidade. "
        "Das profundezas do oceano às vastas extensões do espaço, "
        "cada avanço transforma a nossa compreensão do mundo que nos rodeia."
    ),
    "Spanish": (
        "Bienvenidos a todos, hoy vamos a explorar juntos algo realmente fascinante. "
        "A lo largo de la historia, filósofos y científicos han cuestionado la esencia misma de la existencia, "
        "empujando las fronteras a través de la observación, la experimentación y la pura curiosidad. "
        "Desde las profundidades del océano hasta las vastas extensiones del espacio, "
        "cada descubrimiento transforma nuestra comprensión del mundo que nos rodea."
    ),
    "Italian": (
        "Benvenuti a tutti, oggi esploreremo insieme qualcosa di davvero affascinante. "
        "Nel corso della storia, filosofi e scienziati hanno messo in discussione l'essenza stessa dell'esistenza, "
        "spingendo i confini attraverso l'osservazione, la sperimentazione e la pura curiosità. "
        "Dalle profondità dell'oceano alle vaste distese dello spazio, "
        "ogni scoperta trasforma la nostra comprensione del mondo che ci circonda."
    ),
}

VOICE_THEMES: dict[str, dict] = {
    "narrator-m": {
        "gender": "male",
        "instructs": {
            "Chinese": "男性，35岁，男中音，温暖沉稳，专业播音，吐字清晰",
            "English": "Male, 35 years old, baritone, warm and steady, professional narrator with clear enunciation",
            "Japanese": "男性、35歳、バリトン、温かく安定した声、プロのナレーター、明瞭な発音",
            "Korean": "남성, 35세, 바리톤, 따뜻하고 안정적인 목소리, 전문 내레이터, 명확한 발음",
            "German": "Männlich, 35 Jahre, Bariton, warm und ruhig, professioneller Erzähler mit klarer Aussprache",
            "French": "Homme, 35 ans, baryton, voix chaude et posée, narrateur professionnel avec diction claire",
            "Russian": "Мужчина, 35 лет, баритон, тёплый и уверенный голос, профессиональный диктор, чёткая дикция",
            "Portuguese": "Masculino, 35 anos, barítono, voz quente e firme, narrador profissional com dicção clara",
            "Spanish": "Masculino, 35 años, barítono, voz cálida y firme, narrador profesional con dicción clara",
            "Italian": "Maschile, 35 anni, baritono, voce calda e stabile, narratore professionale con dizione chiara",
        },
    },
    "narrator-f": {
        "gender": "female",
        "instructs": {
            "Chinese": "女性，30岁，女中音，流畅动人，专业播音，自信从容",
            "English": "Female, 30 years old, mezzo-soprano, smooth and engaging, professional narrator with confident delivery",
            "Japanese": "女性、30歳、メゾソプラノ、滑らかで魅力的な声、プロのナレーター、自信のある話し方",
            "Korean": "여성, 30세, 메조소프라노, 부드럽고 매력적인 목소리, 전문 내레이터, 자신감 있는 전달",
            "German": "Weiblich, 30 Jahre, Mezzosopran, geschmeidig und fesselnd, professionelle Erzählerin mit selbstbewusstem Vortrag",
            "French": "Femme, 30 ans, mezzo-soprano, voix fluide et captivante, narratrice professionnelle au débit assuré",
            "Russian": "Женщина, 30 лет, меццо-сопрано, плавный и увлекательный голос, профессиональный диктор, уверенная подача",
            "Portuguese": "Feminino, 30 anos, mezzo-soprano, voz suave e envolvente, narradora profissional com entrega confiante",
            "Spanish": "Femenino, 30 años, mezzosoprano, voz suave y cautivadora, narradora profesional con entrega segura",
            "Italian": "Femminile, 30 anni, mezzosoprano, voce fluida e coinvolgente, narratrice professionale con tono sicuro",
        },
    },
    "young-m": {
        "gender": "male",
        "instructs": {
            "Chinese": "男性，22岁，男高音，充满活力，轻松自然，年轻的对话风格",
            "English": "Male, 22 years old, tenor, energetic and casual, natural youthful conversational tone",
            "Japanese": "男性、22歳、テノール、エネルギッシュでカジュアル、自然で若々しい会話調",
            "Korean": "남성, 22세, 테너, 활기차고 캐주얼한, 자연스럽고 젊은 대화체",
            "German": "Männlich, 22 Jahre, Tenor, energisch und locker, natürlicher jugendlicher Gesprächston",
            "French": "Homme, 22 ans, ténor, énergique et décontracté, ton conversationnel jeune et naturel",
            "Russian": "Мужчина, 22 года, тенор, энергичный и непринуждённый, естественный молодёжный разговорный тон",
            "Portuguese": "Masculino, 22 anos, tenor, enérgico e descontraído, tom conversacional jovem e natural",
            "Spanish": "Masculino, 22 años, tenor, enérgico y relajado, tono conversacional joven y natural",
            "Italian": "Maschile, 22 anni, tenore, energico e disinvolto, tono conversazionale giovane e naturale",
        },
    },
    "young-f": {
        "gender": "female",
        "instructs": {
            "Chinese": "女性，22岁，女高音，明亮开朗，友好亲切，年轻的对话风格",
            "English": "Female, 22 years old, soprano, bright and cheerful, friendly youthful conversational style",
            "Japanese": "女性、22歳、ソプラノ、明るく快活、フレンドリーで若々しい会話スタイル",
            "Korean": "여성, 22세, 소프라노, 밝고 쾌활한, 친근하고 젊은 대화 스타일",
            "German": "Weiblich, 22 Jahre, Sopran, hell und fröhlich, freundlicher jugendlicher Gesprächsstil",
            "French": "Femme, 22 ans, soprano, voix claire et joyeuse, style conversationnel jeune et amical",
            "Russian": "Женщина, 22 года, сопрано, яркий и жизнерадостный голос, дружелюбный молодёжный разговорный стиль",
            "Portuguese": "Feminino, 22 anos, soprano, voz brilhante e alegre, estilo conversacional jovem e amigável",
            "Spanish": "Femenino, 22 años, soprano, voz brillante y alegre, estilo conversacional joven y amigable",
            "Italian": "Femminile, 22 anni, soprano, voce brillante e allegra, stile conversazionale giovane e amichevole",
        },
    },
    "deep-m": {
        "gender": "male",
        "instructs": {
            "Chinese": "男性，45岁，低男中音，深沉有力，威严从容，节奏沉稳",
            "English": "Male, 45 years old, bass-baritone, deep and commanding, authoritative presence with measured pacing",
            "Japanese": "男性、45歳、バスバリトン、深く力強い声、威厳のある存在感、落ち着いたペース",
            "Korean": "남성, 45세, 베이스바리톤, 깊고 위엄 있는 목소리, 권위 있는 존재감, 차분한 속도",
            "German": "Männlich, 45 Jahre, Bassbariton, tief und bestimmend, autoritäre Ausstrahlung mit gemessenem Tempo",
            "French": "Homme, 45 ans, baryton-basse, voix profonde et imposante, présence autoritaire au rythme mesuré",
            "Russian": "Мужчина, 45 лет, бас-баритон, глубокий и властный голос, авторитетное присутствие, размеренный темп",
            "Portuguese": "Masculino, 45 anos, baixo-barítono, voz profunda e imponente, presença autoritária com ritmo comedido",
            "Spanish": "Masculino, 45 años, bajo-barítono, voz profunda e imponente, presencia autoritaria con ritmo mesurado",
            "Italian": "Maschile, 45 anni, basso-baritono, voce profonda e imponente, presenza autorevole con ritmo misurato",
        },
    },
    "deep-f": {
        "gender": "female",
        "instructs": {
            "Chinese": "女性，45岁，女低音，低沉有力，威严沉着，节奏稳健",
            "English": "Female, 45 years old, contralto, deep and resonant, commanding presence with deliberate pacing",
            "Japanese": "女性、45歳、コントラルト、深く響く声、威厳のある存在感、意図的なペース",
            "Korean": "여성, 45세, 콘트랄토, 깊고 울림 있는 목소리, 위엄 있는 존재감, 신중한 속도",
            "German": "Weiblich, 45 Jahre, Kontraalt, tief und resonant, gebietende Ausstrahlung mit bedächtigem Tempo",
            "French": "Femme, 45 ans, contralto, voix profonde et résonnante, présence imposante au rythme délibéré",
            "Russian": "Женщина, 45 лет, контральто, глубокий и звучный голос, властное присутствие, размеренный темп",
            "Portuguese": "Feminino, 45 anos, contralto, voz profunda e ressonante, presença imponente com ritmo deliberado",
            "Spanish": "Femenino, 45 años, contralto, voz profunda y resonante, presencia imponente con ritmo deliberado",
            "Italian": "Femminile, 45 anni, contralto, voce profonda e risonante, presenza imponente con ritmo deliberato",
        },
    },
    "warm-m": {
        "gender": "male",
        "instructs": {
            "Chinese": "男性，38岁，男中音，温暖柔和，亲切安抚，节奏舒缓",
            "English": "Male, 38 years old, baritone, warm and gentle, comforting pace with a reassuring tone",
            "Japanese": "男性、38歳、バリトン、温かく穏やかな声、安心感のあるペース、落ち着いたトーン",
            "Korean": "남성, 38세, 바리톤, 따뜻하고 부드러운 목소리, 편안한 속도, 안심시키는 톤",
            "German": "Männlich, 38 Jahre, Bariton, warm und sanft, beruhigendes Tempo mit tröstlichem Klang",
            "French": "Homme, 38 ans, baryton, voix chaude et douce, rythme apaisant avec un ton rassurant",
            "Russian": "Мужчина, 38 лет, баритон, тёплый и мягкий голос, успокаивающий темп, утешительная интонация",
            "Portuguese": "Masculino, 38 anos, barítono, voz quente e suave, ritmo reconfortante com tom acolhedor",
            "Spanish": "Masculino, 38 años, barítono, voz cálida y suave, ritmo reconfortante con tono acogedor",
            "Italian": "Maschile, 38 anni, baritono, voce calda e gentile, ritmo rassicurante con tono confortante",
        },
    },
    "warm-f": {
        "gender": "female",
        "instructs": {
            "Chinese": "女性，38岁，女低音，温暖舒缓，轻柔安抚，节奏从容",
            "English": "Female, 38 years old, alto, warm and soothing, gentle pace with a comforting tone",
            "Japanese": "女性、38歳、アルト、温かく落ち着いた声、穏やかなペース、安心感のあるトーン",
            "Korean": "여성, 38세, 알토, 따뜻하고 부드러운 목소리, 편안한 속도, 위로하는 톤",
            "German": "Weiblich, 38 Jahre, Alt, warm und beruhigend, sanftes Tempo mit tröstlichem Klang",
            "French": "Femme, 38 ans, alto, voix chaude et apaisante, rythme doux avec un ton réconfortant",
            "Russian": "Женщина, 38 лет, альт, тёплый и успокаивающий голос, мягкий темп, утешительная интонация",
            "Portuguese": "Feminino, 38 anos, contralto, voz quente e reconfortante, ritmo suave com tom acolhedor",
            "Spanish": "Femenino, 38 años, contralto, voz cálida y reconfortante, ritmo suave con tono acogedor",
            "Italian": "Femminile, 38 anni, contralto, voce calda e rassicurante, ritmo dolce con tono confortante",
        },
    },
    "news-m": {
        "gender": "male",
        "instructs": {
            "Chinese": "男性，40岁，男中音，清晰有力，新闻播报风格，权威客观，节奏明快",
            "English": "Male, 40 years old, baritone, crisp and articulate, broadcast news anchor style, authoritative and objective",
            "Japanese": "男性、40歳、バリトン、クリアで明瞭、ニュースキャスター風、権威的で客観的なトーン",
            "Korean": "남성, 40세, 바리톤, 선명하고 명료한, 뉴스 앵커 스타일, 권위적이고 객관적인 톤",
            "German": "Männlich, 40 Jahre, Bariton, klar und deutlich, Nachrichtensprecher-Stil, sachlich und autoritativ",
            "French": "Homme, 40 ans, baryton, voix nette et articulée, style présentateur de journal, ton autoritaire et objectif",
            "Russian": "Мужчина, 40 лет, баритон, чёткий и отчётливый, стиль диктора новостей, авторитетный и объективный тон",
            "Portuguese": "Masculino, 40 anos, barítono, voz nítida e articulada, estilo âncora de telejornal, tom autoritário e objetivo",
            "Spanish": "Masculino, 40 años, barítono, voz nítida y articulada, estilo presentador de noticias, tono autoritario y objetivo",
            "Italian": "Maschile, 40 anni, baritono, voce nitida e articolata, stile giornalista televisivo, tono autorevole e obiettivo",
        },
    },
    "news-f": {
        "gender": "female",
        "instructs": {
            "Chinese": "女性，36岁，女中音，清晰利落，新闻播报风格，沉稳专业，节奏紧凑",
            "English": "Female, 36 years old, mezzo-soprano, clear and polished, broadcast news anchor style, composed and professional",
            "Japanese": "女性、36歳、メゾソプラノ、明瞭で洗練された声、ニュースキャスター風、落ち着きのあるプロの話し方",
            "Korean": "여성, 36세, 메조소프라노, 명확하고 세련된, 뉴스 앵커 스타일, 침착하고 전문적인 전달",
            "German": "Weiblich, 36 Jahre, Mezzosopran, klar und geschliffen, Nachrichtensprecherin-Stil, gelassen und professionell",
            "French": "Femme, 36 ans, mezzo-soprano, voix claire et soignée, style présentatrice de journal, posée et professionnelle",
            "Russian": "Женщина, 36 лет, меццо-сопрано, чёткий и отточенный голос, стиль ведущей новостей, собранная и профессиональная",
            "Portuguese": "Feminino, 36 anos, mezzo-soprano, voz clara e polida, estilo âncora de telejornal, composta e profissional",
            "Spanish": "Femenino, 36 años, mezzosoprano, voz clara y pulida, estilo presentadora de noticias, serena y profesional",
            "Italian": "Femminile, 36 anni, mezzosoprano, voce chiara e curata, stile giornalista televisiva, composta e professionale",
        },
    },
    "storyteller-m": {
        "gender": "male",
        "instructs": {
            "Chinese": "男性，50岁，男中音，富有表现力，戏剧性的抑扬顿挫，引人入胜的讲故事风格",
            "English": "Male, 50 years old, baritone, expressive and dramatic, rich vocal dynamics with captivating storytelling cadence",
            "Japanese": "男性、50歳、バリトン、表現力豊か、ドラマチックな抑揚、引き込まれる語り口",
            "Korean": "남성, 50세, 바리톤, 표현력이 풍부한, 극적인 억양, 매력적인 이야기 전달 스타일",
            "German": "Männlich, 50 Jahre, Bariton, ausdrucksstark und dramatisch, lebhafte Stimmführung mit fesselndem Erzählrhythmus",
            "French": "Homme, 50 ans, baryton, expressif et dramatique, dynamique vocale riche avec une cadence narrative captivante",
            "Russian": "Мужчина, 50 лет, баритон, выразительный и драматичный, богатая голосовая динамика, захватывающая манера повествования",
            "Portuguese": "Masculino, 50 anos, barítono, expressivo e dramático, rica dinâmica vocal com cadência narrativa cativante",
            "Spanish": "Masculino, 50 años, barítono, expresivo y dramático, rica dinámica vocal con cadencia narrativa cautivadora",
            "Italian": "Maschile, 50 anni, baritono, espressivo e drammatico, ricca dinamica vocale con cadenza narrativa avvincente",
        },
    },
    "storyteller-f": {
        "gender": "female",
        "instructs": {
            "Chinese": "女性，48岁，女中音，富有感染力，情感丰富的抑扬顿挫，娓娓道来的叙事风格",
            "English": "Female, 48 years old, mezzo-soprano, expressive and evocative, emotive vocal range with enchanting storytelling rhythm",
            "Japanese": "女性、48歳、メゾソプラノ、表現力豊かで情感的、感動的な声域、魅惑的な語りのリズム",
            "Korean": "여성, 48세, 메조소프라노, 표현력이 풍부하고 감성적인, 감동적인 음역, 매혹적인 이야기 리듬",
            "German": "Weiblich, 48 Jahre, Mezzosopran, ausdrucksstark und stimmungsvoll, emotionaler Stimmumfang mit bezauberndem Erzählrhythmus",
            "French": "Femme, 48 ans, mezzo-soprano, expressive et évocatrice, tessiture émotive avec un rythme narratif envoûtant",
            "Russian": "Женщина, 48 лет, меццо-сопрано, выразительный и проникновенный голос, эмоциональный диапазон, завораживающий ритм повествования",
            "Portuguese": "Feminino, 48 anos, mezzo-soprano, expressiva e evocativa, alcance vocal emotivo com ritmo narrativo encantador",
            "Spanish": "Femenino, 48 años, mezzosoprano, expresiva y evocadora, rango vocal emotivo con ritmo narrativo encantador",
            "Italian": "Femminile, 48 anni, mezzosoprano, espressiva ed evocativa, gamma vocale emotiva con ritmo narrativo incantevole",
        },
    },
    "kid-m": {
        "gender": "male",
        "instructs": {
            "Chinese": "男孩，8岁，童声高音，天真活泼，好奇心强，语速偏快，清脆明亮",
            "English": "Boy, 8 years old, high-pitched child voice, innocent and lively, curious and enthusiastic, bright and clear",
            "Japanese": "男の子、8歳、高い子供の声、無邪気で元気、好奇心旺盛、明るくはっきりした声",
            "Korean": "남자아이, 8세, 높은 어린이 목소리, 순수하고 활발한, 호기심 가득한, 밝고 또렷한 목소리",
            "German": "Junge, 8 Jahre, hohe Kinderstimme, unschuldig und lebhaft, neugierig und begeistert, klar und hell",
            "French": "Garçon, 8 ans, voix aiguë d'enfant, innocent et vif, curieux et enthousiaste, voix claire et lumineuse",
            "Russian": "Мальчик, 8 лет, высокий детский голос, наивный и живой, любознательный и увлечённый, звонкий и чистый",
            "Portuguese": "Menino, 8 anos, voz aguda infantil, inocente e animado, curioso e entusiasmado, voz clara e brilhante",
            "Spanish": "Niño, 8 años, voz aguda infantil, inocente y vivaz, curioso y entusiasta, voz clara y brillante",
            "Italian": "Bambino, 8 anni, voce acuta infantile, innocente e vivace, curioso ed entusiasta, voce chiara e brillante",
        },
    },
    "kid-f": {
        "gender": "female",
        "instructs": {
            "Chinese": "女孩，8岁，童声高音，甜美可爱，活泼好动，语速偏快，清脆悦耳",
            "English": "Girl, 8 years old, high-pitched child voice, sweet and cheerful, playful and animated, crisp and pleasant",
            "Japanese": "女の子、8歳、高い子供の声、可愛らしく明るい、活発で楽しげ、澄んだ心地よい声",
            "Korean": "여자아이, 8세, 높은 어린이 목소리, 사랑스럽고 밝은, 활발하고 즐거운, 맑고 상쾌한 목소리",
            "German": "Mädchen, 8 Jahre, hohe Kinderstimme, süß und fröhlich, verspielt und lebhaft, klar und angenehm",
            "French": "Fille, 8 ans, voix aiguë d'enfant, douce et joyeuse, espiègle et animée, voix nette et agréable",
            "Russian": "Девочка, 8 лет, высокий детский голос, милая и весёлая, игривая и живая, чистый и приятный голос",
            "Portuguese": "Menina, 8 anos, voz aguda infantil, doce e alegre, brincalhona e animada, voz nítida e agradável",
            "Spanish": "Niña, 8 años, voz aguda infantil, dulce y alegre, juguetona y animada, voz nítida y agradable",
            "Italian": "Bambina, 8 anni, voce acuta infantile, dolce e allegra, giocosa e vivace, voce nitida e piacevole",
        },
    },
    "teen-m": {
        "gender": "male",
        "instructs": {
            "Chinese": "男性，16岁，男高音偏高，声音略带沙哑，自信随性，青少年对话风格",
            "English": "Male, 16 years old, higher tenor, slightly husky, confident and laid-back, teenage conversational style",
            "Japanese": "男性、16歳、高めのテノール、やや掠れた声、自信があり気さく、ティーンの会話スタイル",
            "Korean": "남성, 16세, 높은 테너, 약간 허스키한, 자신감 있고 편안한, 십대 대화 스타일",
            "German": "Männlich, 16 Jahre, höherer Tenor, leicht rau, selbstbewusst und lässig, jugendlicher Gesprächsstil",
            "French": "Homme, 16 ans, ténor aigu, voix légèrement rauque, confiant et décontracté, style conversationnel adolescent",
            "Russian": "Юноша, 16 лет, высокий тенор, слегка хрипловатый, уверенный и непринуждённый, подростковый разговорный стиль",
            "Portuguese": "Masculino, 16 anos, tenor mais agudo, levemente rouco, confiante e descontraído, estilo conversacional adolescente",
            "Spanish": "Masculino, 16 años, tenor más agudo, levemente ronco, seguro y relajado, estilo conversacional adolescente",
            "Italian": "Maschile, 16 anni, tenore più acuto, leggermente roco, sicuro e disinvolto, stile conversazionale adolescenziale",
        },
    },
    "teen-f": {
        "gender": "female",
        "instructs": {
            "Chinese": "女性，16岁，女高音，清澈明亮，热情外向，青少年对话风格",
            "English": "Female, 16 years old, soprano, crystal clear and bright, enthusiastic and outgoing, teenage conversational style",
            "Japanese": "女性、16歳、ソプラノ、透き通った明るい声、熱心で社交的、ティーンの会話スタイル",
            "Korean": "여성, 16세, 소프라노, 투명하고 밝은 목소리, 열정적이고 사교적인, 십대 대화 스타일",
            "German": "Weiblich, 16 Jahre, Sopran, kristallklar und hell, begeistert und aufgeschlossen, jugendlicher Gesprächsstil",
            "French": "Femme, 16 ans, soprano, voix cristalline et lumineuse, enthousiaste et extravertie, style conversationnel adolescent",
            "Russian": "Девушка, 16 лет, сопрано, кристально чистый и яркий голос, увлечённая и общительная, подростковый разговорный стиль",
            "Portuguese": "Feminino, 16 anos, soprano, voz cristalina e brilhante, entusiasmada e extrovertida, estilo conversacional adolescente",
            "Spanish": "Femenino, 16 años, soprano, voz cristalina y brillante, entusiasta y extrovertida, estilo conversacional adolescente",
            "Italian": "Femminile, 16 anni, soprano, voce cristallina e brillante, entusiasta e socievole, stile conversazionale adolescenziale",
        },
    },
}


def list_themes() -> list[dict]:
    return [
        {"name": name, "gender": t["gender"], "languages": sorted(t["instructs"])}
        for name, t in VOICE_THEMES.items()
    ]


def _theme_profile_name(theme_name: str, language: str, gender: str) -> str:
    """Build the HuggingFace profile directory name for a theme."""
    return f"general-{language}-{gender}-{theme_name}"


def _try_download_theme_profile(
    profile_name: str,
    cache_dir: str,
) -> tuple[str, str] | None:
    """Try to download a pre-generated theme profile from HuggingFace.

    Returns ``(wav_path, script_path)`` on success, or ``None`` if the
    remote profile does not exist.
    """
    profile_dir = os.path.join(cache_dir, "themes", profile_name)
    wav_path = os.path.join(profile_dir, "voice.wav")
    script_path = os.path.join(profile_dir, SCRIPT_FILENAME)

    # Already downloaded previously
    if (
        os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0
        and os.path.isfile(script_path)
    ):
        log.info("Using cached remote theme profile: %s", profile_dir)
        return wav_path, script_path

    base_url = f"{PROFILES_REPO_URL}/general/{profile_name}"

    try:
        _download_file(f"{base_url}/{SCRIPT_FILENAME}", script_path)
        _download_file(f"{base_url}/voice.wav", wav_path)
        log.info("Downloaded theme profile from HuggingFace: %s", profile_name)
        return wav_path, script_path
    except HTTPError:
        # Clean up partial downloads
        for p in (wav_path, script_path):
            if os.path.exists(p):
                os.remove(p)
        log.info(
            "Theme profile %s not found on HuggingFace, will generate locally",
            profile_name,
        )
        return None


def resolve_theme(
    theme_name: str,
    language: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    cache_dir: str | None = None,
) -> tuple[str, str]:
    """Return ``(voice_sample_path, ref_text)`` for a theme + language pair.

    Resolution order:

    1. **Local cache** — reuse a previously downloaded or generated file.
    2. **HuggingFace** — download the pre-generated profile from
       ``profiles/general/general-{lang}-{gender}-{theme}/``.
    3. **Local generation** — synthesise with the Qwen3-TTS VoiceDesign
       model and cache the result.
    """
    if theme_name not in VOICE_THEMES:
        raise ValueError(
            f"Unknown voice theme {theme_name!r}. "
            f"Available: {', '.join(VOICE_THEMES)}"
        )
    theme = VOICE_THEMES[theme_name]
    if language not in theme["instructs"]:
        raise ValueError(
            f"Language {language!r} not supported for theme {theme_name!r}. "
            f"Available: {', '.join(sorted(theme['instructs']))}"
        )

    instruct = theme["instructs"][language]
    ref_text = _THEME_REF_TEXTS[language]

    if cache_dir is None:
        cache_dir = os.path.join(
            tempfile.gettempdir(), "mazinger-dubber-profiles"
        )

    # ── 1. Check local cache ────────────────────────────────────────
    theme_dir = os.path.join(cache_dir, "themes", theme_name)
    wav_path = os.path.join(theme_dir, f"{language}.wav")

    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
        log.info("Using cached voice theme: %s", wav_path)
        return wav_path, ref_text

    # ── 2. Try downloading pre-generated profile from HuggingFace ───
    hf_name = _theme_profile_name(theme_name, language, theme["gender"])
    result = _try_download_theme_profile(hf_name, cache_dir)
    if result is not None:
        hf_wav, _ = result
        # Copy into the standard theme cache location for future reuse
        os.makedirs(theme_dir, exist_ok=True)
        import shutil
        shutil.copy2(hf_wav, wav_path)
        log.info("Theme profile cached from HuggingFace: %s", wav_path)
        return wav_path, ref_text

    # ── 3. Fall back to local generation ────────────────────────────
    from mazinger.tts import design_voice
    import soundfile as sf

    log.info("Generating voice for theme %s / %s", theme_name, language)
    audio, sr = design_voice(
        text=ref_text, language=language, instruct=instruct,
        device=device, dtype=dtype,
    )
    os.makedirs(theme_dir, exist_ok=True)
    sf.write(wav_path, audio, sr)
    log.info("Voice theme cached: %s", wav_path)
    return wav_path, ref_text


def generate_profile(
    theme_name: str,
    language: str,
    output_dir: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
) -> tuple[str, str]:
    """Generate a reusable profile directory from a voice theme.

    Writes ``voice.wav`` and ``script.txt`` into *output_dir* so the
    directory can later be passed as ``--clone-profile <path>``.

    Returns:
        ``(voice_wav_path, script_txt_path)``
    """
    wav_path, ref_text = resolve_theme(
        theme_name, language, device=device, dtype=dtype,
    )

    os.makedirs(output_dir, exist_ok=True)
    dest_wav = os.path.join(output_dir, "voice.wav")
    dest_script = os.path.join(output_dir, SCRIPT_FILENAME)

    import shutil
    shutil.copy2(wav_path, dest_wav)
    with open(dest_script, "w", encoding="utf-8") as fh:
        fh.write(ref_text)

    log.info("Profile written to %s", output_dir)
    return dest_wav, dest_script


def _load_local_profile(profile_dir: str) -> tuple[str, str]:
    """Load voice sample and script from a local profile directory."""
    script_path = os.path.join(profile_dir, SCRIPT_FILENAME)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"No {SCRIPT_FILENAME} in {profile_dir}")

    for ext in VOICE_EXTENSIONS:
        voice_path = os.path.join(profile_dir, f"voice.{ext}")
        if os.path.isfile(voice_path):
            return _ensure_wav(voice_path), script_path

    raise FileNotFoundError(
        f"No voice file in {profile_dir} "
        f"(tried: {', '.join(f'voice.{e}' for e in VOICE_EXTENSIONS)})"
    )


def create_auto_clone_profile(
    audio_path: str,
    srt_path: str,
    output_dir: str,
) -> str:
    """Create a voice profile by extracting a segment from *audio_path*.

    Selects a contiguous block of SRT entries spanning 20–60 s with the
    highest word count, extracts the matching audio slice, and writes
    ``voice.wav`` into *output_dir*.

    Returns:
        Path to the extracted ``voice.wav``.
    """
    from mazinger.srt import parse_file

    entries = parse_file(srt_path)
    if not entries:
        raise ValueError("Cannot auto-clone: SRT file is empty")

    total_dur = entries[-1]["end"] - entries[0]["start"]
    if total_dur < MIN_CLONE_DURATION:
        raise ValueError(
            f"Cannot auto-clone: source audio is {total_dur:.1f}s "
            f"(need at least {MIN_CLONE_DURATION:.0f}s). "
            "Provide --voice-sample + --voice-script instead."
        )

    best_i = best_j = 0
    best_words = 0

    for i in range(len(entries)):
        words = 0
        for j in range(i, len(entries)):
            span = entries[j]["end"] - entries[i]["start"]
            words += len(entries[j]["text"].split())
            if span > MAX_CLONE_DURATION:
                break
            if span >= MIN_CLONE_DURATION and words > best_words:
                best_words = words
                best_i, best_j = i, j

    if best_words < MIN_CLONE_WORDS:
        raise ValueError(
            f"Cannot auto-clone: best segment has {best_words} words "
            f"(need at least {MIN_CLONE_WORDS}). "
            "Provide --voice-sample + --voice-script instead."
        )

    start_t = entries[best_i]["start"]
    end_t = entries[best_j]["end"]

    os.makedirs(output_dir, exist_ok=True)
    wav_path = os.path.join(output_dir, "voice.wav")

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start_t), "-to", str(end_t),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            wav_path,
        ],
        check=True, capture_output=True,
    )

    log.info(
        "Auto-cloned voice profile: %.1fs, %d words -> %s",
        end_t - start_t, best_words, output_dir,
    )
    return wav_path


def fetch_profile(profile_name: str, cache_dir: str | None = None) -> tuple[str, str]:
    """Return ``(voice_sample_path, voice_script_path)`` for *profile_name*.

    *profile_name* can be:

    - A **local directory path** containing ``voice.wav`` and ``script.txt``
      (e.g. generated by :func:`generate_profile`).
    - A **remote profile name** from the HuggingFace dataset (e.g. ``abubakr``).

    Remote files are downloaded once into *cache_dir* (default: a persistent
    temp directory) and reused on subsequent calls.

    Returns:
        A ``(voice_sample, voice_script)`` tuple of local file paths.
        The voice sample is always a WAV file.
    """
    if os.path.isdir(profile_name):
        return _load_local_profile(profile_name)

    def _profile_exists(base_url: str) -> bool:
        url = f"{base_url}/{profile_name}/{SCRIPT_FILENAME}"
        try:
            urllib.request.urlopen(_make_request(url, "HEAD"), timeout=10)
            return True
        except Exception:
            return False

    default_url = DEFAULT_PROFILES_REPO_URL
    custom_url = CUSTOM_PROFILES_REPO_URL

    in_default = _profile_exists(default_url)
    in_custom = _profile_exists(custom_url) if custom_url else False

    if in_default and in_custom:
        if sys.stdin.isatty():
            print(f"Profile '{profile_name}' found in both repos:")
            print(f"  [1] Default: {default_url}")
            print(f"  [2] Custom: {custom_url}")
            choice = input("Select repo (1/2): ").strip()
            base_url = custom_url if choice == "2" else default_url
        else:
            raise ValueError(
                f"Profile '{profile_name}' exists in both default and custom repos. "
                f"Run interactively to select, or unset MAZINGER_PROFILES_REPO_URL."
            )
    elif in_custom:
        base_url = custom_url
    elif in_default:
        base_url = default_url
    else:
        raise FileNotFoundError(f"Profile '{profile_name}' not found")

    if cache_dir is None:
        cache_dir = os.path.join(
            tempfile.gettempdir(), "mazinger-dubber-profiles"
        )

    profile_dir = os.path.join(cache_dir, profile_name)
    base = f"{base_url}/{profile_name}"

    # Download script
    script_path = os.path.join(profile_dir, SCRIPT_FILENAME)
    if not os.path.exists(script_path):
        _download_file(f"{base}/{SCRIPT_FILENAME}", script_path)
    else:
        log.info("Using cached script: %s", script_path)

    # Download voice (try wav/m4a/mp3) and convert to WAV
    raw_voice = _download_voice(base, profile_dir)
    voice_path = _ensure_wav(raw_voice)

    return voice_path, script_path
