CXX = g++
CXXFLAGS = --std=c++2a -Wall -Wextra -Wpedantic -g -O3 -Isrc/games/minichess -Isrc/state -Isrc

# 自動抓取 policy 下所有的 .cpp，這樣鴨子就會被包進去
POLICY_SRCS = $(wildcard src/policy/*.cpp)

GAME_SRCS = src/games/minichess/state.cpp
UBGI_SRCS = src/ubgi/ubgi.cpp
BENCH_SRCS = src/benchmark.cpp
TEST_SRCS = unittest/state_test.cpp

all: minichess benchmark state_test

minichess:
	$(CXX) $(CXXFLAGS) -o build/minichess-ubgi $(GAME_SRCS) $(POLICY_SRCS) $(UBGI_SRCS)

benchmark:
	$(CXX) $(CXXFLAGS) -o build/minichess-benchmark $(GAME_SRCS) $(POLICY_SRCS) $(BENCH_SRCS)

state_test:
	$(CXX) $(CXXFLAGS) -o unittest/build/state_test $(GAME_SRCS) $(POLICY_SRCS) $(TEST_SRCS)

# Windows 安全版 clean：只刪除自己編譯出來的目標檔案，不動 TA 的 baseline
clean:
	-del /Q /F build\minichess-ubgi.exe 2>nul
	-del /Q /F build\minichess-benchmark.exe 2>nul
	-del /Q /F unittest\build\state_test.exe 2>nul