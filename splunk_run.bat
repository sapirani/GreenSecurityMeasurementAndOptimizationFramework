@echo [off]
python scanner.py
"C:\Program Files\Splunk\bin\splunk.exe" stop
"C:\Program Files\Splunk\bin\splunk.exe" clean eventdata -index eventgen -f
pause