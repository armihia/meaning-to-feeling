from .prompts import EMOTIONS

LABELS = [
    "高兴",
    "悲伤",
    "愤怒",
    "恐惧",
    "惊讶",
    "焦虑",
    "受挫",
    "困惑",
    "内疚",
    "平淡",
]
COLORS = [
    "#d69428",
    "#597db2",
    "#c86458",
    "#8375af",
    "#c181a2",
    "#bc8a5f",
    "#9d735e",
    "#6b94a0",
    "#978ba0",
    "#81958a",
]
EXAMPLES = [
    dict(
        id="stayed",
        name="意外的支持",
        category="理解转折",
        text="我以为这次又要一个人收拾残局，没想到大家都留了下来。",
        context_text="小组临近截止时间还没完成任务，前几次都是我独自留下来处理。",
        target_character="我",
    ),
    dict(
        id="door",
        name="说不出口的失落",
        category="表里不同",
        text="我说没关系，关上门后却坐了很久。",
        context_text="我为这次见面准备了很久，朋友临出门时说他不来了。",
        target_character="我",
    ),
    dict(
        id="praise",
        name="同一句评价 · 夸奖",
        category="语境对照",
        text="我听见他说“真有你的”，一时没有接话。",
        context_text="设备突然坏了，我临时想出办法，让大家顺利完成了演出。他笑着走过来。",
        target_character="我",
    ),
    dict(
        id="irony",
        name="同一句评价 · 责怪",
        category="语境对照",
        text="我听见他说“真有你的”，一时没有接话。",
        context_text="我没有检查就提交了文件，害得大家的成果被覆盖。他沉着脸走过来。",
        target_character="我",
    ),
    dict(
        id="stage",
        name="站到台前",
        category="感受变化",
        text="我站到台前，手心发热，开口后反而平静了。",
        context_text="这是我第一次向一百多人介绍自己的研究，已经准备了两周。",
        target_character="我",
    ),
    dict(
        id="message",
        name="落空的机会",
        category="新信息",
        text="我点开那条消息，看到这个机会最后给了别人。",
        context_text="我为这次选拔投入了几个月，一直觉得自己很有希望。",
        target_character="我",
    ),
]
PROFILES = [
    dict(
        id="shared",
        name="共享模型",
        source="shared",
        source_label="无个人校准",
        description="先看同一句话的共享结果",
        profile=None,
    ),
    dict(
        id="demo_positive",
        name="演示 · 积极偏移",
        source="illustrative",
        source_label="手动演示",
        description="提高高兴分量，降低平淡分量",
        profile=[0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.12, 0.0, 0.0],
    ),
    dict(
        id="demo_fine",
        name="演示 · 细分倾向",
        source="illustrative",
        source_label="手动演示",
        description="保持情绪偏移为零，提高切分密度",
        profile=[0.0] * 10 + [0.12, 0.08],
    ),
]
HISTORY_EXAMPLE = dict(
    label_source="illustrative",
    emotion_order=EMOTIONS,
    history=[
        dict(
            sample_id="example-1",
            text="我终于完成了任务。",
            context_text="这项工作已经持续了一周。",
            target_character="我",
            emotion_observations=[
                dict(
                    end_char=9,
                    values=[0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4],
                )
            ],
            boundary_observations=[],
        )
    ],
)
