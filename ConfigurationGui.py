import tkinter as tk
from configurations import *
from Scanner import main
class APP:

    def __init__(self):
        def print_answers():
            global power_plan
            global scan_option
            global scan_type
            global chosen_power_plan_name
            global chosen_power_plan_guid
            global MINIMUM_DELTA_CAPACITY
            global MINIMUM_SCAN_TIME
            power_plan = power_plan_inside.get()
            scan_option = scan_mode_inside.get()
            scan_type = scan_type_inside.get()
            chosen_power_plan_name = power_plan[0]
            chosen_power_plan_guid = power_plan[1]
            MINIMUM_DELTA_CAPACITY = e1.get()
            MINIMUM_SCAN_TIME = e2.get()* MINUTE
            main()
        def props(cls):   
            return [i for i in cls.__dict__.keys() if i[:1] != '_']
        self.win = tk.Tk()
        self.win.geometry("400x400")
        self.win.title("Green Security - Scanner configuration")
        self.win.configure(bg='#E3DCA8')
        power_plan_options = props(PowerPlan)
        scan_mode_options = props(ScanMode)
        scan_type_options = props(ScanType)

        # Variable to keep track of the option
        # selected in OptionMenu
        power_plan_inside = tk.StringVar(self.win)
        scan_mode_inside = tk.StringVar(self.win)
        scan_type_inside = tk.StringVar(self.win)
        
        power_plan_inside.set("Select an Option")
        tk.Label(self.win, 
         text="power_plan").grid(row=0, column=0)
        question_menu_power_plan = tk.OptionMenu(self.win, power_plan_inside, *power_plan_options)
        question_menu_power_plan.grid(row=0, column=1)
        
        scan_mode_inside.set("Select an Option")
        tk.Label(self.win, 
         text="scan mode").grid(row=1, column=0)
        question_menu_scan_mode = tk.OptionMenu(self.win, scan_mode_inside, *scan_mode_options)
        question_menu_scan_mode.grid(row=1, column=1)
        
        scan_type_inside.set("Select an Option")
        tk.Label(self.win, 
         text="scan type").grid(row=2, column=0)
        question_menu_scan_type = tk.OptionMenu(self.win, scan_type_inside, *scan_type_options)
        question_menu_scan_type.grid(row=2, column=1)
        
        tk.Label(self.win, 
         text="MINIMUM_DELTA_CAPACITY").grid(row=3, column=0)
        e1 = tk.Entry(self.win)
        e1.grid(row=3, column=1)
        tk.Label(self.win, 
                text="MINIMUM_SCAN_TIME").grid(row=4, column=0)
        e2 = tk.Entry(self.win)
        e2.grid(row=4, column=1)


        # Submit button
        # Whenever we click the submit button, our submitted
        # option is printed ---Testing purpose
        submit_button = tk.Button(self.win, text='Scan', command=print_answers)
        submit_button.grid(row=5, column=1)
        self.win.mainloop()



if __name__ == '__main__':
    APP()
