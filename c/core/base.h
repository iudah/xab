#ifndef XAB_BASE_H
#define XAB_BASE_H

#if defined _WIN32 && !STANDALONE

#ifdef XABEXPORTS
#define XABAPI __declspec(dllexport)
#else
#define XABAPI __declspec(dllimport)
#endif
#define XABCALL __cdecl

#else

#define XABAPI
#define XABCALL

#endif

#endif