SOURCE  = *.java algorithms/*.java problems/*.java
CLASSES = *.class algorithms/*.class problems/*.class
JARFILE = javann.jar

all: classes

classes:
	javac $(SOURCE)

jar: classes
	jar -cvfm $(JARFILE) MANIFEST.MF $(CLASSES) && mv $(JARFILE) ..

clean:
	rm -f $(CLASSES)

# vim:noet
