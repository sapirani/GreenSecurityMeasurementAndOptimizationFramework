@echo off
python scanner.py

set "output_dir=C:\Users\Administrator\Repositories\GreenSecurity-FirstExperiment\Dell Inc. Latitude Latitude 7430 Windows 10\Splunk Enterprise SIEM\Power Saver Plan\One Scan"

REM Find the newest directory in the output directory
for /f "delims=" %%d in ('dir /ad /b /o-d "%output_dir%"') do (
    set "newest_dir=%output_dir%\%%~d"
    goto :done
)
:done

"C:\Program Files\Splunk\bin\splunk.exe" search "index=eventgen" -output csv -maxout 20000000 > "%newest_dir%\output.csv" -auth shoueii:sH231294
timeout /t 120
"C:\Program Files\Splunk\bin\splunk.exe" stop
"C:\Program Files\Splunk\bin\splunk.exe" clean eventdata -index eventgen -f
pause


