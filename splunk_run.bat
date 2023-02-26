@echo [off]
python scanner.py
"C:\Program Files\Splunk\bin\splunk.exe" stop
"C:\Program Files\Splunk\bin\splunk.exe" clean eventdata -index eventgen -f
"C:\Program Files\Splunk\bin\splunk.exe" search "index=eventgen" -output csv -maxout 20000 > C:\Users\Administrator\Repositories\output.csv  
pause