# -*- coding: utf-8 -*-
"""
数据结构定义：ImageInfo / MaskInfo / Explanation / Sample。
"""
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ImageInfo:
    path: str
    height: int
    width: int


@dataclass
class MaskInfo:
    path: str
    class_id: int
    role: str = "main"  # main / aux


@dataclass
class Explanation:
    function: str
    structure: str
    context: str


@dataclass
class Sample:
    image_id: str
    image: ImageInfo
    masks: List[MaskInfo]
    task_type: str
    language: str
    question: str
    class_id: int
    class_name: str
    explanation: Optional[Explanation] = None
    answer: Optional[str] = None
