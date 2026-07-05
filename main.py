import os
from colorama import init, Fore, Style
from agents import Agent

init(autoreset=True)  # makes colors work properly on Windows

# --- Create specialist agents ---

researcher = Agent(
    name="Researcher",
    system_prompt="You are a research specialist. You gather and explain factual information clearly and concisely. When relevant, use web search to find current, accurate information.",
    use_web_search=True
)

writer = Agent(
    name="Writer",
    system_prompt="You are a writing specialist. You take information and turn it into clear, well-organized, polished text."
)

planner = Agent(
    name="Planner",
    system_prompt="You are a planning specialist. You break big goals or tasks into clear, numbered, actionable steps."
)

coder = Agent(
    name="Coder",
    system_prompt="You are a coding specialist. You write clean, working code and explain it simply, assuming the reader is a beginner unless told otherwise."
)

agents = {
    "researcher": researcher,
    "writer": writer,
    "planner": planner,
    "coder": coder,
}

AGENT_COLORS = {
    "researcher": Fore.CYAN,
    "writer": Fore.GREEN,
    "planner": Fore.YELLOW,
    "coder": Fore.MAGENTA,
}

# --- Manager: decides which agent(s) should handle the task, in order ---

def router(user_task):
    routing_prompt = f"""A user gave this task: "{user_task}"

Decide which specialist agent(s) should handle this, IN ORDER, comma-separated.
Available agents:
- researcher — facts, information, current events, explanations
- writer — turning info into polished written text
- planner — breaking a task into steps
- coder — writing or explaining code

Examples:
- "What is nitrogen fixation?" -> researcher
- "Write a bio for me" -> writer
- "Plan and code a budget tracker" -> planner, coder
- "Research soil erosion and write a summary" -> researcher, writer

Reply with ONLY the agent name(s), comma-separated, nothing else."""

    decision = researcher.run(routing_prompt).strip().lower()
    chain = [name.strip() for name in decision.split(",") if name.strip() in agents]
    return chain if chain else ["researcher"]


def run_task(user_task):
    chain = router(user_task)
    print(f"\n[Manager decided: {' -> '.join(chain)}]\n")

    context = user_task
    final_result = None

    for step_num, agent_name in enumerate(chain, start=1):
        agent = agents[agent_name]
        color = AGENT_COLORS[agent_name]

        print(f"{color}--- Step {step_num}: {agent.name} ---{Style.RESET_ALL}")
        result = agent.run(context)
        print(f"{color}{result}{Style.RESET_ALL}\n")

        # feed this result as context into the next agent in the chain
        context = f"Original task: {user_task}\n\nPrevious agent's output:\n{result}\n\nContinue the task."
        final_result = result

        # if the writer was involved, save its output to a file
        if agent_name == "writer":
            save_writer_output(result)

    return final_result


def save_writer_output(text):
    os.makedirs("outputs", exist_ok=True)
    existing = len(os.listdir("outputs"))
    filename = f"outputs/writer_output_{existing + 1}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{Fore.BLUE}[Saved to {filename}]{Style.RESET_ALL}\n")


# --- Continuous loop ---
if __name__ == "__main__":
    print(f"{Fore.WHITE}{Style.BRIGHT}Your AI agent ecosystem is ready. Type 'exit' to quit.{Style.RESET_ALL}\n")

    while True:
        task = input("What do you need help with? ")

        if task.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        run_task(task)
        print("-" * 40 + "\n")