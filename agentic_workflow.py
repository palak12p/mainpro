import json
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Dict, List

from data_retriever import get_user_context, retrieve_data
from rbac import detect_role


ClassifierFn = Callable[[str], Awaitable[Dict[str, str]]]
ChatFn = Callable[[str, str, str], Awaitable[str]]


@dataclass
class AgentStep:
    agent: str
    action: str
    status: str
    detail: str


@dataclass
class AgentResult:
    answer: str
    role: str
    classification: Dict[str, str]
    user_context: Dict[str, Any]
    trace: List[Dict[str, str]]


def _step(agent: str, action: str, status: str, detail: str) -> Dict[str, str]:
    return asdict(AgentStep(agent=agent, action=action, status=status, detail=detail))


def _compact_payload(retrieved: Dict[str, Any]) -> Dict[str, Any]:
    records = retrieved.get("records", [])
    return {
        "intent": retrieved.get("intent"),
        "entity": retrieved.get("entity"),
        "summary": retrieved.get("summary", {}),
        "record_count": len(records),
        "sample_records": records[:12],
        "requester_context": retrieved.get("requester_context", {}),
    }


def _structured_answer(query: str, role: str, payload: Dict[str, Any]) -> str:
    intent = payload.get("intent", "general")
    summary = payload.get("summary", {})
    entity = payload.get("entity", "general")

    if intent == "student_count":
        return f"There are {summary.get('count', 0)} students in {summary.get('course', entity)}."

    if intent == "course_enrollment":
        students = summary.get("students", [])
        names = ", ".join(student.get("name", "") for student in students[:8] if student.get("name"))
        suffix = "..." if summary.get("count", 0) > 8 else ""
        return (
            f"{summary.get('count', 0)} students are enrolled in {summary.get('course', entity)}. "
            f"Sample list: {names}{suffix}"
        )

    if intent == "faculty_list":
        faculty = ", ".join(summary.get("faculty", [])[:10])
        return f"Faculty linked to {summary.get('course', entity)}: {faculty or 'No faculty data found.'}"

    if intent == "grades_average":
        if "average_cgpa" in summary:
            return (
                f"The average performance for {summary.get('course', entity)} is "
                f"{summary.get('average_grade_point', 0)} grade points with an average CGPA of "
                f"{summary.get('average_cgpa', 0)}."
            )
        averages = summary.get("course_averages", [])
        parts = [
            f"{item.get('course')}: CGPA {item.get('cgpa', 0)}, grade points {item.get('grade_points', 0)}"
            for item in averages[:6]
        ]
        return "Course-wise averages: " + "; ".join(parts)

    if intent == "attendance_report":
        return (
            f"The average attendance for {summary.get('course', entity)} is "
            f"{summary.get('average_attendance_percent', 0)}% across "
            f"{summary.get('student_count', 0)} students."
        )

    if intent == "student_profile":
        return (
            f"{summary.get('name')} ({summary.get('student_id')}) is in {summary.get('department')}, "
            f"semester {summary.get('semester')}, section {summary.get('section')}. "
            f"CGPA: {summary.get('cgpa')}, attendance: {summary.get('attendance_percent')}%, "
            f"mentor: {summary.get('mentor', {}).get('name')}, class teacher: "
            f"{summary.get('class_teacher', {}).get('name')}."
        )

    if intent == "mentor_lookup":
        mentor = summary.get("mentor", {})
        return (
            f"The mentor for {summary.get('name')} is {mentor.get('name')}. "
            f"Email: {mentor.get('email')}, phone: {mentor.get('phone')}."
        )

    if intent == "class_teacher_info":
        teacher = summary.get("class_teacher", {})
        return (
            f"{summary.get('name')} is in section {summary.get('section')}. "
            f"The class teacher is {teacher.get('name')}. "
            f"Email: {teacher.get('email')}, phone: {teacher.get('phone')}."
        )

    if intent == "backlog_report":
        students = summary.get("students", [])
        if not students:
            return "No backlog students were found in the current scope."
        rows = [
            f"{student.get('name')} ({student.get('student_id')}): {student.get('backlog_count')} backlog(s)"
            for student in students[:10]
        ]
        return (
            f"{summary.get('count_with_backlogs', 0)} students currently have backlogs. "
            f"Examples: {'; '.join(rows)}"
        )

    if intent == "contact_lookup":
        mentor = summary.get("mentor_contact", {})
        teacher = summary.get("class_teacher_contact", {})
        student = summary.get("student_contact", {})
        return (
            f"Contact details for {summary.get('name')}: student phone {student.get('phone')}, "
            f"college email {student.get('college_email')}. Mentor: {mentor.get('name')} "
            f"({mentor.get('email')}, {mentor.get('phone')}). Class teacher: {teacher.get('name')} "
            f"({teacher.get('email')}, {teacher.get('phone')})."
        )

    return f"I found academic data for your query: {query}"


async def run_agentic_workflow(
    *,
    user_id: str,
    query: str,
    data_path: str,
    assignments_path: str,
    classify_query: ClassifierFn,
    ask_groq: ChatFn,
) -> AgentResult:
    trace: List[Dict[str, str]] = []
    role = detect_role(user_id)
    trace.append(_step("role-guard-agent", "detect_role", "completed", f"Resolved user role as {role}."))

    user_context = get_user_context(
        data_path=data_path,
        user_id=user_id,
        role=role,
        assignments_path=assignments_path,
    )
    trace.append(_step("context-agent", "load_user_context", "completed", "Loaded role-scoped dashboard context."))

    classification = await classify_query(query)
    trace.append(
        _step(
            "router-agent",
            "classify_query",
            "completed",
            f"Routed query as {classification.get('query_type')} with intent {classification.get('intent')}.",
        )
    )

    query_type = classification.get("query_type", "general_query")
    intent = classification.get("intent", "general")
    entity = classification.get("entity", "general")

    if query_type == "general_query":
        trace.append(_step("knowledge-agent", "answer_general_query", "in_progress", "Sending conceptual query to Groq."))
        answer = await ask_groq(
            "You are an educational assistant. Answer clearly, directly, and accurately.",
            f"User role: {role}\nQuestion: {query}",
            "llama-3.3-70b-versatile",
        )
        trace[-1]["status"] = "completed"
        trace[-1]["detail"] = "Returned direct LLM answer for general knowledge query."
        return AgentResult(
            answer=answer,
            role=role,
            classification=classification,
            user_context=user_context,
            trace=trace,
        )

    trace.append(_step("data-agent", "retrieve_academic_data", "in_progress", f"Fetching academic data for intent {intent}."))
    retrieved = retrieve_data(
        data_path=data_path,
        intent=intent,
        entity=entity,
        role=role,
        user_id=user_id,
        assignments_path=assignments_path,
    )
    trace[-1]["status"] = "completed"
    trace[-1]["detail"] = "Academic records loaded and summarized."

    compact_payload = _compact_payload(retrieved)

    if intent in {
        "student_count",
        "course_enrollment",
        "faculty_list",
        "grades_average",
        "attendance_report",
        "student_profile",
        "mentor_lookup",
        "class_teacher_info",
        "backlog_report",
        "contact_lookup",
    }:
        trace.append(_step("executor-agent", "compose_structured_answer", "completed", "Generated direct tool-based response."))
        answer = _structured_answer(query, role, compact_payload)
    else:
        trace.append(_step("composer-agent", "compose_natural_language_response", "in_progress", "Sending compact tool output to Groq composer."))
        answer = await ask_groq(
            (
                "You are Moodle AI Assistant for NMIT. "
                "Use the provided structured tool output to answer clearly. "
                "Prefer the structured summary. Keep it concise and role-aware."
            ),
            (
                f"User role: {role}\n"
                f"Original query: {query}\n"
                f"Classification: {json.dumps(classification)}\n"
                f"Tool output: {json.dumps(compact_payload)}\n"
                "Write the final answer."
            ),
            "llama-3.3-70b-versatile",
        )
        trace[-1]["status"] = "completed"
        trace[-1]["detail"] = "Natural-language answer composed from tool results."

    return AgentResult(
        answer=answer,
        role=role,
        classification=classification,
        user_context=user_context,
        trace=trace,
    )
