# Save
torch.save(model.state_dict(), "test%d.pth"%testnum)

# Load later
# got to have the model class definition.
#model = MyModelClass(...)     # define your model architecture
#model.load_state_dict(torch.load("model_weights.pth"))
#model.eval()
