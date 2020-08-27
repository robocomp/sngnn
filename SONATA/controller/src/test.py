from contributor_GUI import *

app = QApplication(sys.argv)
form = CForm()
form.show()
app.exec_()
var = form.val
print("--",var)
