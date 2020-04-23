from ai.models import MemoryNetwork


model = MemoryNetwork()

print(model.forward(["Hi there", "hmmm"], [["Hello", "How are you?"], ["Hello", "How are you?"]], ["Hello", "How are you?"]))