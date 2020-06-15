---
name: Help request / Bug report
about: Ask for help with a problem
title: Single line summary of the problem here
labels: question
assignees: ''

---

**Describe the problem**
A clear and concise description of what the problem is.

**To Reproduce (if applicable)**
Steps to reproduce the behavior. Include a short, self-contained code section which will cause the problem. These marks (```python) make code blocks readable.

```python
import tomopy

if __name__ == "__main__":
    tomopy.recon()  # error happens here!
```

```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: recon() missing 2 required positional arguments: 'tomo' and 'theta'
```

**Expected behavior (if applicable)**
A clear and concise description of what you expected to happen.

**Helpful images (if applicable)**
Add screenshots to help explain your problem.

**Platform Information:**
 - OS: [e.g. macOS, Windows 10, Ubuntu]
 - Python Version [e.g. 3.5]
 - TomoPy Version [e.g. 1.5.0]

**Additional context**
Add any other context about the problem here.
