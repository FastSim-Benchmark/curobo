"""URDF chain utilities for the standalone planner experiments."""
from __future__ import annotations

from collections.abc import Iterable
from xml.etree.ElementTree import Element


def needed_joints(document: Element, links: Iterable[str], base: str) -> set[str]:
    """Return movable joints on the requested link chains below ``base``."""
    parents = {joint.find("child").get("link"): joint for joint in document.findall("joint")}
    names: set[str] = set()
    for tip in links:
        visited: set[str] = set()
        while tip != base:
            if tip in visited or tip not in parents:
                raise ValueError(f"URDF chain does not reach configured base: {tip}")
            visited.add(tip)
            joint = parents[tip]
            if joint.get("type") != "fixed":
                names.add(joint.get("name"))
            tip = joint.find("parent").get("link")
    return names
