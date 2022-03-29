
@echo off

cd gym_microrts\microrts

for /r .\src\ %%i in (*.java) do (echo Y | javac -d .\build -cp .\lib\* -sourcepath .\src %%i)

cd build
for /r .\ %%i in (*.jar) do (
    echo adding dependency %%i
    jar xf %%i
)

jar cvf microrts.jar *
echo Y | move microrts.jar ..\microrts.jar
cd ..
echo Y | DEL /S build