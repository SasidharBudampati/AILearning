from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel()

agent = CodeAgent(tools=[], model=model)

result = agent.run("Write python code to print a 3d grid without for loop")

print(result)