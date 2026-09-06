"""Small, dependency-free geometry editing helpers used by Studio."""

from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np


class GeometryExpressionError(ValueError):
    pass


SHAPE_CHOICES = ("Circle", "Triangle", "Square", "Regular polygon")


def shape_choice(kind: str, sides: int | None = None) -> str:
    """Return the human editor choice for the canonical motif representation."""
    if kind == "circle":
        return "Circle"
    if kind != "polygon":
        raise GeometryExpressionError(f"unsupported motif kind: {kind}")
    if int(sides or 0) == 3:
        return "Triangle"
    if int(sides or 0) == 4:
        return "Square"
    return "Regular polygon"


def canonical_shape(choice: str, sides: Any = None) -> tuple[str, int]:
    """Map a controlled UI choice to the compact scientific geometry schema."""
    if choice == "Circle":
        return "circle", 0
    if choice == "Triangle":
        return "polygon", 3
    if choice == "Square":
        return "polygon", 4
    if choice == "Regular polygon":
        count = int(safe_number(sides))
        if count < 3:
            raise GeometryExpressionError("regular polygon must have at least 3 sides")
        return "polygon", count
    raise GeometryExpressionError("choose Circle, Triangle, Square, or Regular polygon")


def safe_number(value: Any) -> float:
    """Parse numeric editor text without eval or arbitrary names/calls."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
    else:
        text = str(value).strip()
        if not text:
            raise GeometryExpressionError("numeric value is required")
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise GeometryExpressionError("invalid numeric expression") from exc
        def evaluate(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            if isinstance(node, ast.Name) and node.id == "pi":
                return math.pi
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                number = evaluate(node.operand)
                return number if isinstance(node.op, ast.UAdd) else -number
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Add): return left + right
                if isinstance(node.op, ast.Sub): return left - right
                if isinstance(node.op, ast.Mult): return left * right
                if isinstance(node.op, ast.Div): return left / right
                return left ** right
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1 and not node.keywords:
                return math.sqrt(evaluate(node.args[0]))
            raise GeometryExpressionError("allowed syntax: numbers, pi, sqrt(), +, -, *, /, **, and parentheses")
        try:
            result = evaluate(tree)
        except (ArithmeticError, OverflowError, ValueError) as exc:
            raise GeometryExpressionError("numeric expression is not finite") from exc
    if not math.isfinite(result):
        raise GeometryExpressionError("numeric value must be finite")
    return result


def epsilon_from_editor(value: Any, representation: str) -> float:
    number = safe_number(value)
    if representation == "n":
        number = number * number
    if number <= 0 or not math.isfinite(number):
        raise GeometryExpressionError("epsilon/n must be positive and finite")
    return number


def editor_value(epsilon: float, representation: str) -> float:
    epsilon = float(epsilon)
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise GeometryExpressionError("epsilon must be positive and finite")
    return math.sqrt(epsilon) if representation == "n" else epsilon


def uniaxial_matrix(factor: Any, angle_degrees: Any) -> np.ndarray:
    factor = safe_number(factor)
    angle = math.radians(safe_number(angle_degrees))
    if factor <= 0:
        raise GeometryExpressionError("uniaxial factor must be positive")
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return rotation @ np.diag([factor, 1.0]) @ rotation.T


def model_name(lattice: str, motifs: list[dict[str, Any]]) -> str:
    motif = motifs[0].get("kind", "geometry") if motifs else "empty"
    return f"{lattice.title()}{motif.title()}"
