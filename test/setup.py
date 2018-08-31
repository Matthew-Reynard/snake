from setuptools import setup

setup(
	name="test",
	version="1.0",
	description="Snake game",
	author="Matthew Reynard",
	author_email="matthewreynard24@gmail.com",
	url="matthewreynard.com",
	# packages=["snake"],
	py_modules=[
	"snakeAI",
	"foodAI",
	"obstacleAI",
	"SnakeGame",
	"QLearn"
	],
	install_requires=[
	"numpy",
	"pygame",
	"pandas"
	]
)