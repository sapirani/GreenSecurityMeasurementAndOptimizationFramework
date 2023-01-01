import os
import time
import win32com.client
import random
import string
import pyautogui
from abc import abstractmethod
import pywinauto.keyboard as keyboard

class AbstractTask:
    def __init__():
        pass

    @abstractmethod
    def run_task(self):
        pass

    def get_name(self):
        return self.__class__.__name__

# around 6 minutes     
class Word(AbstractTask):

    def __init__(self):
        # super(Word, self).__init__()        
        pass
    def run_task(self):
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = True
        
        doc = word.Documents.Add()
        paragraph = doc.Paragraphs[0]
        paragraph.Range.Text = "This is a test document created using Python and pywin32."
        
        for i in range(20):
            time.sleep(10)
            paragraph = doc.Paragraphs(doc.Paragraphs.Count).Range
            paragraph.InsertParagraphAfter()
            paragraph.InsertAfter('This is a new paragraph.')
            time.sleep(10)
        
        doc.SaveAs(os.path.expanduser("~/word_test{}.docx".format(random.randint(1, 1000))))
        print('word saved')
        time.sleep(10)    
        doc.Close()
        word.Quit()  
 
# around 9 minutes    
class Excel(AbstractTask):

    def __init__(self):
        # super(Excel, self).__init__()
        pass
        
    def run_task(self):
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = True
        workbook = excel.Workbooks.Add()

        worksheet = workbook.Worksheets[0]
        name_list = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hannah", "Ivan", "Judy", "Kevin", "Linda", "Mike", "Nancy", "Oscar", "Pam", "Quinn", "Ralph", "Sally", "Tom", "Ursula", "Victor", "Wendy", "Xavier", "Yvonne", "Zach"]
        age_list = [random.randint(18, 65) for i in range(100)]
        
        worksheet.Range("A1").Value = "Name"
        worksheet.Range("B1").Value = "Age"
        for i in range(2, 100):
            worksheet.Range("A{}".format(i)).Value = [random.choice(name_list) for i in range(len(name_list))]
            worksheet.Range("B{}".format(i)).Value = [random.randint(18, 65) for i in range(100)]
            time.sleep(5)
        
        time.sleep(30)
        workbook.SaveAs(os.path.expanduser("~/excel_test{}.xlsx".format(random.randint(1, 1000))))
        # print('excel saved')
        
        workbook.Close()
        excel.Quit()  

# around 30 seconds
class PowerPoint(AbstractTask):

    def __init__(self):
        # super(PowerPoint, self).__init__()
        pass
        

    def run_task(self):
        ppt = win32com.client.Dispatch("PowerPoint.Application")
        ppt.Visible = True
        
        # presentation = ppt.Presentations.Add()
        # slide = presentation.Slides.Add(1, win32com.client.constants.ppLayoutTitleOnly)
        # shape = slide.Shapes[0]
        # shape.TextFrame.TextRange.Text = "This is a test slide created using Python and pywin32."
        # time.sleep(10)
        
        # presentation.SaveAs(os.path.expanduser("~/test.pptx"))
        # presentation.Close()
        time.sleep(30)
        ppt.quit()

# around 30 seconds       
class Outlook(AbstractTask):

    def __init__(self):
        # super(Outlook, self).__init__()
        pass
    def run_task(self):
        outlook = win32com.client.Dispatch("Outlook.Application")

        # message = outlook.CreateItem(0)
        # message.To = "recipient@example.com"
        # message.Subject = "Test Email"
        # message.Body = "This is a test email sent using Python and pywin32."
        # message.Send()
        time.sleep(30)
        outlook.Quit()
   