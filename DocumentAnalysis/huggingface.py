from langchain_huggingface import HuggingFaceEndpoint

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

examples = [
    {
        "question": """Are these two requirements similar? Answer \"Yes\" or \"No\"
        DAFMAN Requirement:
        8.14.2.  The software shall use secure methods like root-of-trust and cryptographic secure boot 
        to ensure that the weapon system executes only trusted applications. (T-1) Verification 
        activities should prove that the weapon system does not execute any untrusted software prior 
        to trusted validation.
        CERT C++ Requirement:
        Rule Chunk 7:
        2.6 DCL55-CPP. Avoid information leakage when passing a class 
        object across a trust boundary 
        The C++ Standard, [class.mem], paragraph 13 [ISO/IEC 14882-2014], describes the layout of 
        non-static data members of a non-union class, specifying the following: 
        Nonstatic data members of a (non-union) class with the same access control are 
        allocated so that later members have higher addresses within a class object. The order 
        of allocation of non-static data members with different access control is unspecified. 
        Implementation alignment requirements might cause two adjacent members not to be 
        allocated immediately after each other; so might requirements for space for managing 
        virtual functions and virtual base classes. 
        Further, [class.bit], paragraph 1, in part, states the following: 
        Allocation of bit-fields within a class object is implementation-defined. Alignment of bit-
        fields is implementation-defined. Bit-fields are packed into some addressable allocation 
        unit. 
        Thus, padding bits may be present at any location within a class object instance (including at the 
        beginning of the object, in the case of an unnamed bit-field as the first member declared in a 
        class). Unless initialized by zero-initialization, padding bits contain indeterminate values that may 
        contain sensitive information. 
        When passing a pointer to a class object instance across a trust boundary to a different trusted 
        domain, the programmer must ensure that the padding bits of such an object do not contain 
        sensitive information. 
        Declarations and Initialization (DCL) - DCL55-CPP. Avoid information leakage when passing a class object across a trust 
        boundary 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        42 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
        2.6.1 
        Noncompliant Code Example 
        This noncompliant code example runs in kernel space and copies data from arg to user space. 
        However, padding bits may be used within the object, for example, to ensure the proper alignment 
        of class data members. These padding bits may contain sensitive information that may then be 
        leaked when the data is copied to user space, regardless of how the data is copied. 
        #include <cstddef> 
        
        struct test { 
        int a; 
        char b; 
        int c; 
        }; 
        
        // Safely copy bytes to user space 
        extern int copy_to_user(void *dest, void *src, std::size_t size); 
        
        void do_stuff(void *usr_buf) { 
        test arg{1, 2, 3}; 
        copy_to_user(usr_buf, &arg, sizeof(arg)); 
        } 
        Declarations and Initialization (DCL) - DCL55-CPP. Avoid information leakage when passing a class object across a trust 
        boundary 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        43 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
        2.6.2 
        Noncompliant Code Example 
        In this noncompliant code example, arg is value-initialized through direct initialization. Because 
        test does not have a user-provided default constructor, the value-initialization is preceded by a 
        zero-initialization that guarantees the padding bits are initialized to 0 before any further 
        initialization occurs. It is akin to using std::memset() to initialize all of the bits in the object 
        to 0. 
        #include <cstddef> 
        
        struct test { 
        int a; 
        char b; 
        int c; 
        }; 
        
        // Safely copy bytes to user space 
        extern int copy_to_user(void *dest, void *src, std::size_t size); 
        
        void do_stuff(void *usr_buf) { 
        test arg{}; 
        
        arg.a = 1; 
        arg.b = 2; 
        arg.c = 3; 
        
        copy_to_user(usr_buf, &arg, sizeof(arg)); 
        } 
        However, compilers are free to implement arg.b = 2 by setting the low byte of a 32-bit 
        register to 2, leaving the high bytes unchanged, and storing all 32 bits of the register into memory. 
        This could leak the high-order bytes resident in the register to a user. 
        Declarations and Initialization (DCL) - DCL55-CPP. Avoid information leakage when passing a class object across a trust 
        boundary 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        44 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
        2.6.3 
        Compliant Solution 
        This compliant solution serializes the structure data before copying it to an untrusted context. 
        #include <cstddef> 
        #include <cstring> 
        
        struct test { 
        int a; 
        char b; 
        int c; 
        }; 
        
        // Safely copy bytes to user space. 
        extern int copy_to_user(void *dest, void *src, std::size_t size); 
        
        void do_stuff(void *usr_buf) { 
        test arg{1, 2, 3}; 
        // May be larger than strictly needed. 
        unsigned char buf[sizeof(arg)]; 
        std::size_t offset = 0; 
        
        std::memcpy(buf + offset, &arg.a, sizeof(arg.a)); 
        offset += sizeof(arg.a); 
        std::memcpy(buf + offset, &arg.b, sizeof(arg.b)); 
        offset += sizeof(arg.b); 
        std::memcpy(buf + offset, &arg.c, sizeof(arg.c)); 
        offset += sizeof(arg.c); 
        
        copy_to_user(usr_buf, buf, offset /* size of info copied */); 
        } 
        This code ensures that no uninitialized padding bits are copied to unprivileged users. The structure 
        copied to user space is now a packed structure and the copy_to_user() function would need 
        to unpack it to recreate the original, padded structure. 
        Declarations and Initialization (DCL) - DCL55-CPP. Avoid information leakage when passing a class object across a trust 
        boundary 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        45 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
        2.6.4 
        Compliant Solution (Padding Bytes) 
        Padding bits can be explicitly declared as fields within the structure. This solution is not portable, 
        however, because it depends on the implementation and target memory architecture. The 
        following solution is specific to the x86-32 architecture. 
        #include <cstddef> 
        
        struct test { 
        int a; 
        char b; 
        char padding_1, padding_2, padding_3; 
        int c; 
        
        test(int a, char b, int c) : a(a), b(b), 
            padding_1(0), padding_2(0), padding_3(0), 
            c(c) {} 
        }; 
        // Ensure c is the next byte after the last padding byte. 
        static_assert(offsetof(test, c) == offsetof(test, padding_3) + 1, 
                    "Object contains intermediate padding"); 
        // Ensure there is no trailing padding. 
        static_assert(sizeof(test) == offsetof(test, c) + sizeof(int), 
                    "Object contains trailing padding"); 
        
        
        
        // Safely copy bytes to user space. 
        extern int copy_to_user(void *dest, void *src, std::size_t size); 
        
        void do_stuff(void *usr_buf) { 
        test arg{1, 2, 3}; 
        copy_to_user(usr_buf, &arg, sizeof(arg)); 
        } 
        The static_assert() declaration accepts a constant expression and an error message. The 
        expression is evaluated at compile time and, if false, the compilation is terminated and the error 
        message is used as the diagnostic. The explicit insertion of the padding bytes into the struct 
        should ensure that no additional padding bytes are added by the compiler, and consequently both 
        static assertions should be true. However, it is necessary to validate these assumptions to ensure 
        that the solution is correct for a particular implementation. 
        2.6.5 
        Noncompliant Code Example 
        In this noncompliant code example, padding bits may abound, including 
        • 
        alignment padding bits after a virtual method table or virtual base class data to align a 
        subsequent data member, 
        Declarations and Initialization (DCL) - DCL55-CPP. Avoid information leakage when passing a class object across a trust 
        boundary 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        46 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
        • 
        alignment padding bits to position a subsequent data member on a properly aligned boundary, 
        • 
        alignment padding bits to position data members of varying access control levels. 
        • 
        bit-field padding bits when the sequential set of bit-fields does not fill an entire allocation 
        unit, 
        • 
        bit-field padding bits when two adjacent bit-fields are declared with different underlying 
        types, 
        • 
        padding bits when a bit-field is declared with a length greater than the number of bits in the 
        underlying allocation unit, or 
        • 
        padding bits to ensure a class instance will be appropriately aligned for use within an array. 
        This code example runs in kernel space and copies data from arg to user space. However, the 
        padding bits within the object instance may contain sensitive information that will then be leaked 
        when the data is copied to user space. 
        #include <cstddef> 
        
        class base { 
        public: 
        virtual ~base() = default; 
        }; 
        
        class test : public virtual base { 
        alignas(32) double h; 
        char i; 
        unsigned j : 80; 
        protected: 
        unsigned k; 
        unsigned l : 4; 
        unsigned short m : 3; 
        public: 
        char n; 
        double o; 
        
        test(double h, char i, unsigned j, unsigned k, unsigned l,  
            unsigned short m, char n, double o) : 
            h(h), i(i), j(j), k(k), l(l), m(m), n(n), o(o) {} 
        
        virtual void foo(); 
        }; 
        
        // Safely copy bytes to user space. 
        extern int copy_to_user(void *dest, void *src, std::size_t size); 
        
        void do_stuff(void *usr_buf) { 
        test arg{0.0, 1, 2, 3, 4, 5, 6, 7.0}; 
        copy_to_user(usr_buf, &arg, sizeof(arg)); 
        } 
        Declarations and Initialization (DCL) - DCL55-CPP. Avoid information leakage when passing a class object across a trust 
        boundary 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        47 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
        Padding bits are implementation-defined, so the layout of the class object may differ between 
        compilers or architectures. When compiled with GCC 5.3.0 for x86-32, the test object requires 
        96 bytes of storage to accommodate 29 bytes of data (33 bytes including the vtable) and has the 
        following layout. 
        Offset (bytes 
        (bits)) 
        Storage Size  
        (bytes (bits)) 
        Reason 
        Offset 
        Storage Size 
        Reason 
        0 
        1 (32) 
        vtable pointer 
        56 (448) 
        4 (32) 
        unsigned k 
        4 (32) 
        28 (224) 
        data member  
        alignment padding 
        60 (480) 
        0 (4) 
        unsigned l : 4 
        32 (256) 
        8 (64) 
        double h 
        60 (484) 
        0 (3) 
        unsigned short m : 3 
        40 (320) 
        1 (8) 
        char i 
        60 (487) 
        0 (1) 
        unused bit-field bits 
        41 (328) 
        3 (24) 
        data member  
        alignment padding 
        61 (488) 
        1 (8) 
        char n 
        44 (352) 
        4 (32) 
        unsigned j : 
        80 
        62 (496) 
        2 (16) 
        data member alignment 
        padding 
        48 (384) 
        6 (48) 
        extended bit-field 
        size padding 
        64 (512) 
        8 (64) 
        double o 
        54 (432) 
        2 (16) 
        alignment padding 
        72 (576) 
        24 (192) 
        class alignment padding 
        Declarations and Initialization (DCL) - DCL55-CPP. Avoid information leakage when passing a class object across a trust 
        boundary 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        48 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
        2.6.6 
        Compliant Solution 
        Due to the complexity of the data structure, this compliant solution serializes the object data 
        before copying it to an untrusted context instead of attempting to account for all of the padding 
        bytes manually. 
        #include <cstddef> 
        #include <cstring> 
        
        class base { 
        public: 
        virtual ~base() = default; 
        }; 
        class test : public virtual base { 
        alignas(32) double h; 
        char i; 
        unsigned j : 80; 
        protected: 
        unsigned k; 
        unsigned l : 4; 
        unsigned short m : 3; 
        public: 
        char n; 
        double o; 
        
        test(double h, char i, unsigned j, unsigned k, unsigned l, 
            unsigned short m, char n, double o) : 
            h(h), i(i), j(j), k(k), l(l), m(m), n(n), o(o) {} 
        
        virtual void foo(); 
        bool serialize(unsigned char *buffer, std::size_t &size) { 
            if (size < sizeof(test)) { 
            return false; 
            } 
            
            std::size_t offset = 0; 
            std::memcpy(buffer + offset, &h, sizeof(h)); 
            offset += sizeof(h); 
            std::memcpy(buffer + offset, &i, sizeof(i)); 
            offset += sizeof(i); 
        
            // Only sizeof(unsigned) bits are valid, so the following is 
            // not narrowing. 
            unsigned loc_j = j;  
        Declarations and Initialization (DCL) - DCL55-CPP. Avoid information leakage when passing a class object across a trust 
        boundary 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        49 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
            std::memcpy(buffer + offset, &loc_j, sizeof(loc_j)); 
            offset += sizeof(loc_j); 
            std::memcpy(buffer + offset, &k, sizeof(k)); 
            offset += sizeof(k); 
            unsigned char loc_l = l & 0b1111; 
            std::memcpy(buffer + offset, &loc_l, sizeof(loc_l)); 
            offset += sizeof(loc_l); 
            unsigned short loc_m = m & 0b111; 
            std::memcpy(buffer + offset, &loc_m, sizeof(loc_m)); 
            offset += sizeof(loc_m); 
            std::memcpy(buffer + offset, &n, sizeof(n)); 
            offset += sizeof(n); 
            std::memcpy(buffer + offset, &o, sizeof(o)); 
            offset += sizeof(o); 
            
            size -= offset; 
            return true; 
        } 
        }; 
        
        // Safely copy bytes to user space. 
        extern int copy_to_user(void *dest, void *src, size_t size); 
        
        void do_stuff(void *usr_buf) { 
        test arg{0.0, 1, 2, 3, 4, 5, 6, 7.0}; 
        
        // May be larger than strictly needed, will be updated by 
        // calling serialize() to the size of the buffer remaining. 
        std::size_t size = sizeof(arg); 
        unsigned char buf[sizeof(arg)]; 
        if (arg.serialize(buf, size)) { 
            copy_to_user(usr_buf, buf, sizeof(test) - size); 
        } else { 
            // Handle error 
        } 
        } 
        This code ensures that no uninitialized padding bits are copied to unprivileged users. The structure 
        copied to user space is now a packed structure and the copy_to_user() function would need 
        to unpack it to re-create the original, padded structure. 
        Declarations and Initialization (DCL) - DCL55-CPP. Avoid information leakage when passing a class object across a trust 
        boundary 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        50 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
        2.6.7 
        Risk Assessment 
        Padding bits might inadvertently contain sensitive data such as pointers to kernel data structures 
        or passwords. A pointer to such a structure could be passed to other functions, causing 
        information leakage. 
        Rule 
        Severity 
        Likelihood 
        Remediation Cost 
        Priority 
        Level 
        DCL55-CPP 
        Low 
        Unlikely 
        High 
        P1 
        L3 
        2.6.7.1 Related Vulnerabilities 
        Numerous vulnerabilities in the Linux Kernel have resulted from violations of this rule.  
        CVE-2010-4083 describes a vulnerability in which the semctl() system call allows 
        unprivileged users to read uninitialized kernel stack memory because various fields of a 
        semid_ds struct declared on the stack are not altered or zeroed before being copied back to 
        the user.  
        CVE-2010-3881 describes a vulnerability in which structure padding and reserved fields in 
        certain data structures in QEMU-KVM were not initialized properly before being copied to user 
        space. A privileged host user with access to /dev/kvm could use this flaw to leak kernel stack 
        memory to user space.  
        CVE-2010-3477 describes a kernel information leak in act_police where incorrectly 
        initialized structures in the traffic-control dump code may allow the disclosure of kernel memory 
        to user space applications. 
        2.6.8 
        Related Guidelines 
        SEI CERT C Coding Standard 
        DCL39-C. Avoid information leakage when 
        passing a structure across a trust boundary 
        2.6.9 
        Bibliography 
        [ISO/IEC 14882-2014] 
        Subclause 8.5, “Initializers” 
        Subclause 9.2, “Class Members” 
        Subclause 9.6, “Bit-fields” 
        Declarations and Initialization (DCL) - DCL56-CPP. Avoid cycles during initialization of static objects 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        51 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution.
        """,
        "answer": "No",
    },
    {
        "question": """Are these two requirements similar? Answer \"Yes\" or \"No\"
        DAFMAN Requirement:
        8.16.3.  The software shall implement mechanisms to prevent or resolve deadlock conditions 
        if the software is responsible for cancelling a Critical Signal after the withdrawal of human 
        intent or if the software is responsible for issuing a Safing Command. (T-1) Verification 
        activities should prove that the software implements mechanisms such as priority inversion or 
        task-assigned resources to prevent or resolve deadlock conditions.
        CERT C++ Requirement:
        11.2 CON51-CPP. Ensure actively held locks are released on 
        exceptional conditions 
        Mutexes that are used to protect accesses to shared data may be locked using the lock() 
        member function and unlocked using the unlock() member function. If an exception occurs 
        between the call to lock() and the call to unlock(), and the exception changes control flow 
        such that unlock() is not called, the mutex will be left in the locked state and no critical 
        sections protected by that mutex will be allowed to execute. This is likely to lead to deadlock. 
        The throwing of an exception must not allow a mutex to remain locked indefinitely. If a mutex 
        was locked and an exception occurs within the critical section protected by that mutex, the mutex 
        must be unlocked as part of exception handling before rethrowing the exception or continuing 
        execution unless subsequent control flow will unlock the mutex. 
        C++ supplies the lock classes lock_guard, unique_lock, and shared_lock, which can 
        be initialized with a mutex. In its constructor, the lock object locks the mutex, and in its 
        destructor, it unlocks the mutex. The lock_guard class provides a simple RAII wrapper around 
        a mutex. The unique_lock and shared_lock classes also use RAII and provide additional 
        functionality, such as manual control over the locking strategy. The unique_lock class 
        prevents the lock from being copied, although it allows the lock ownership to be moved to another 
        lock. The shared_lock class allows the mutex to be shared by several locks. For all three 
        classes, if an exception occurs and takes control flow out of the scope of the lock, the destructor 
        will unlock the mutex and the program can continue working normally. These lock objects are the 
        preferred way to ensure that a mutex is properly released when an exception is thrown. 
        11.2.1 Noncompliant Code Example 
        This noncompliant code example manipulates shared data and protects the critical section by 
        locking the mutex. When it is finished, it unlocks the mutex. However, if an exception occurs 
        while manipulating the shared data, the mutex will remain locked. 
        #include <mutex> 
        
        void manipulate_shared_data(std::mutex &pm) { 
        pm.lock(); 
        
        // Perform work on shared data. 
        
        pm.unlock(); 
        } 
        Concurrency (CON) - CON51-CPP. Ensure actively held locks are released on exceptional conditions 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        369 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
        11.2.2 Compliant Solution (Manual Unlock) 
        This compliant solution catches any exceptions thrown when performing work on the shared data 
        and unlocks the mutex before rethrowing the exception. 
        #include <mutex> 
        
        void manipulate_shared_data(std::mutex &pm) { 
        pm.lock(); 
        try { 
            // Perform work on shared data. 
        } catch (...) { 
            pm.unlock(); 
            throw; 
        } 
        pm.unlock(); // in case no exceptions occur 
        } 
        11.2.3 Compliant Solution (Lock Object) 
        This compliant solution uses a lock_guard object to ensure that the mutex will be unlocked, 
        even if an exception occurs, without relying on exception handling machinery and manual 
        resource management. 
        #include <mutex> 
        
        void manipulate_shared_data(std::mutex &pm) { 
        std::lock_guard<std::mutex> lk(pm); 
        
        // Perform work on shared data. 
        } 
        11.2.4 Risk Assessment 
        If an exception occurs while a mutex is locked, deadlock may result. 
        Rule 
        Severity 
        Likelihood 
        Remediation Cost 
        Priority 
        Level 
        CON51-CPP 
        Low 
        Probable 
        Low 
        P6 
        L2 
        Concurrency (CON) - CON51-CPP. Ensure actively held locks are released on exceptional conditions 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        370 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
        11.2.5 Related Guidelines 
        This rule is a subset of ERR56-CPP. Guarantee exception safety. 
        MITRE CWE 
        CWE-667, Improper Locking 
        SEI CERT Oracle Coding Standard for Java 
        LCK08-J. Ensure actively held locks are re-
        leased on exceptional conditions 
        11.2.6 Bibliography 
        [ISO/IEC 14882-2014] 
        Subclause 30.4.2, “Locks” 
        Concurrency (CON) - CON52-CPP. Prevent data races when accessing bit-fields from multiple threads 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        371 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution.
        Rule Chunk 75:
        11.3 CON52-CPP. Prevent data races when accessing bit-fields from 
        multiple threads 
        When accessing a bit-field, a thread may inadvertently access a separate bit-field in adjacent 
        memory. This is because compilers are required to store multiple adjacent bit-fields in one storage 
        unit whenever they fit. Consequently, data races may exist not just on a bit-field accessed by 
        multiple threads but also on other bit-fields sharing the same byte or word. The problem is 
        difficult to diagnose because it may not be obvious that the same memory location is being 
        modified by multiple threads. 
        One approach for preventing data races in concurrent programming is to use a mutex. When 
        properly observed by all threads, a mutex can provide safe and secure access to a shared object. 
        However, mutexes provide no guarantees with regard to other objects that might be accessed 
        when the mutex is not controlled by the accessing thread. Unfortunately, there is no portable way 
        to determine which adjacent bit-fields may be stored along with the desired bit-field. 
        Another approach is to insert a non-bit-field member between any two bit-fields to ensure that 
        each bit-field is the only one accessed within its storage unit. This technique effectively 
        guarantees that no two bit-fields are accessed simultaneously. 
        11.3.1 Noncompliant Code Example (bit-field) 
        Adjacent bit-fields may be stored in a single memory location. Consequently, modifying adjacent 
        bit-fields in different threads is undefined behavior, as shown in this noncompliant code example. 
        struct MultiThreadedFlags { 
        unsigned int flag1 : 2; 
        unsigned int flag2 : 2; 
        }; 
        
        MultiThreadedFlags flags; 
        
        void thread1() { 
        flags.flag1 = 1; 
        } 
        
        void thread2() { 
        flags.flag2 = 2; 
        } 
        Concurrency (CON) - CON52-CPP. Prevent data races when accessing bit-fields from multiple threads 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        372 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
        For example, the following instruction sequence is possible. 
        Thread 1: register 0 = flags 
        Thread 1: register 0 &= ~mask(flag1) 
        Thread 2: register 0 = flags 
        Thread 2: register 0 &= ~mask(flag2) 
        Thread 1: register 0 |= 1 << shift(flag1) 
        Thread 1: flags = register 0 
        Thread 2: register 0 |= 2 << shift(flag2) 
        Thread 2: flags = register 0 
        11.3.2 Compliant Solution (bit-field, C++11 and later, mutex) 
        This compliant solution protects all accesses of the flags with a mutex, thereby preventing any 
        data races. 
        #include <mutex> 
        
        struct MultiThreadedFlags { 
        unsigned int flag1 : 2; 
        unsigned int flag2 : 2; 
        }; 
        
        struct MtfMutex { 
        MultiThreadedFlags s; 
        std::mutex mutex; 
        }; 
        
        MtfMutex flags; 
        
        void thread1() { 
        std::lock_guard<std::mutex> lk(flags.mutex); 
        flags.s.flag1 = 1; 
        } 
        
        void thread2() { 
        std::lock_guard<std::mutex> lk(flags.mutex); 
        flags.s.flag2 = 2; 
        } 
        Concurrency (CON) - CON52-CPP. Prevent data races when accessing bit-fields from multiple threads 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        373 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
        11.3.3 Compliant Solution (C++11) 
        In this compliant solution, two threads simultaneously modify two distinct non-bit-field members 
        of a structure. Because the members occupy different bytes in memory, no concurrency protection 
        is required. 
        struct MultiThreadedFlags { 
        unsigned char flag1; 
        unsigned char flag2; 
        }; 
        
        MultiThreadedFlags flags; 
        
        void thread1() { 
        flags.flag1 = 1; 
        } 
        
        void thread2() { 
        flags.flag2 = 2; 
        } 
        Unlike earlier versions of the standard, C++11 and later explicitly define a memory location and 
        provide the following note in [intro.memory] paragraph 4 [ISO/IEC 14882-2014]: 
        [Note: Thus a bit-field and an adjacent non-bit-field are in separate memory locations, 
        and therefore can be concurrently updated by two threads of execution without 
        interference.  The same applies to two bit-fields, if one is declared inside a nested struct 
        declaration and the other is not, or if the two are separated by a zero-length bit-field 
        declaration, or if they are separated by a non-bit-field declaration.  It is not safe to 
        concurrently update two bit-fields in the same struct if all fields between them are also 
        bit-fields of non-zero width.  – end note ] 
        It is almost certain that flag1 and flag2 are stored in the same word. Using a compiler that 
        conforms to earlier versions of the standard, if both assignments occur on a thread-scheduling 
        interleaving that ends with both stores occurring after one another, it is possible that only one of 
        the flags will be set as intended, and the other flag will contain its previous value because both 
        members are represented by the same word, which is the smallest unit the processor can work on. 
        Before the changes made to the C++ Standard for C++11, there were no guarantees that these 
        flags could be modified concurrently. 
        Concurrency (CON) - CON52-CPP. Prevent data races when accessing bit-fields from multiple threads 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        374 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
        11.3.4 Risk Assessment 
        Although the race window is narrow, an assignment or an expression can evaluate improperly 
        because of misinterpreted data resulting in a corrupted running state or unintended information 
        disclosure. 
        Rule 
        Severity 
        Likelihood 
        Remediation Cost 
        Priority 
        Level 
        CON52-CPP 
        Medium 
        Probable 
        Medium 
        P8 
        L2 
        11.3.5 Related Guidelines 
        SEI CERT C Coding Standard 
        CON32-C. Prevent data races when accessing 
        bit-fields from multiple threads 
        11.3.6 Bibliography 
        [ISO/IEC 14882-2014] 
        Subclause 1.7, “The C++ memory model” 
        Concurrency (CON) - CON53-CPP. Avoid deadlock by locking in a predefined order 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        375 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution.
        """,
        "answer": "Yes",
    },
    {
        "question": """Are these two requirements similar? Answer \"Yes\" or \"No\"
        AFMAN Requirement:
        8.15.5.  The software shall not execute memory allocation functions after application 
        initialization. (T-1) Verification activities should prove that the software only executes 
        memory allocation functions before or during application initialization.
        CERT C++ Requirement:
        Rule Chunk 16:
        3.4 EXP53-CPP. Do not read uninitialized memory 
        Local, automatic variables assume unexpected values if they are read before they are initialized. 
        The C++ Standard, [dcl.init], paragraph 12 [ISO/IEC 14882-2014], states the following:  
        If no initializer is specified for an object, the object is default-initialized. When storage for 
        an object with automatic or dynamic storage duration is obtained, the object has an 
        indeterminate value, and if no initialization is performed for the object, that object retains 
        an indeterminate value until that value is replaced. If an indeterminate value is produced 
        by an evaluation, the behavior is undefined except in the following cases: 
        • 
        If an indeterminate value of unsigned narrow character type is produced by the 
        evaluation of: 
            — the second or third operand of a conditional expression, 
            — the right operand of a comma expression, 
            — the operand of a cast or conversion to an unsigned narrow character type, or 
            — a discarded-value expression, 
        then the result of the operation is an indeterminate value. 
        • 
        If an indeterminate value of unsigned narrow character type is produced by the 
        evaluation of the right operand of a simple assignment operator whose first operand 
        is an lvalue of unsigned narrow character type, an indeterminate value replaces the 
        value of the object referred to by the left operand. 
        • 
        If an indeterminate value of unsigned narrow character type is produced by the 
        evaluation of the initialization expression when initializing an object of unsigned 
        narrow character type, that object is initialized to an indeterminate value. 
        The default initialization of an object is described by paragraph 7 of the same subclause: 
        To default-initialize an object of type T means: 
        • 
        if T is a (possibly cv-qualified) class type, the default constructor for T is called (and 
        the initialization is ill-formed if T has no default constructor or overload resolution 
        results in an ambiguity or in a function that is deleted or inaccessible from the 
        context of the initialization); 
        • 
        if T is an array type, each element is default-initialized; 
        • 
        otherwise, no initialization is performed. 
        If a program calls for the default initialization of an object of a const-qualified type T, T 
        shall be a class type with a user-provided default constructor. 
        Expressions (EXP) - EXP53-CPP. Do not read uninitialized memory 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        97 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
        As a result, objects of type T with automatic or dynamic storage duration must be explicitly 
        initialized before having their value read as part of an expression unless T is a class type or an 
        array thereof or is an unsigned narrow character type. If T is an unsigned narrow character type, it 
        may be used to initialize an object of unsigned narrow character type, which results in both 
        objects having an indeterminate value. This technique can be used to implement copy operations 
        such as std::memcpy() without triggering undefined behavior. 
        Additionally, memory dynamically allocated with a new expression is default-initialized when the 
        new-initialized is omitted. Memory allocated by the standard library function std::calloc() 
        is zero-initialized. Memory allocated by the standard library function std::realloc() 
        assumes the values of the original pointer but may not initialize the full range of memory. 
        Memory allocated by any other means (std::malloc(), allocator objects, operator 
        new(), and so on) is assumed to be default-initialized. 
        Objects of static or thread storage duration are zero-initialized before any other initialization takes 
        place [ISO/IEC 14882-2014] and need not be explicitly initialized before having their value read. 
        Reading uninitialized variables for creating entropy is problematic because these memory 
        accesses can be removed by compiler optimization. VU#925211 is an example of a vulnerability 
        caused by this coding error [VU#925211]. 
        3.4.1 
        Noncompliant Code Example 
        In this noncompliant code example, an uninitialized local variable is evaluated as part of an 
        expression to print its value, resulting in undefined behavior. 
        #include <iostream> 
        
        void f() { 
        int i; 
        std::cout << i; 
        } 
        3.4.2 
        Compliant Solution 
        In this compliant solution, the object is initialized prior to printing its value. 
        #include <iostream> 
        
        void f() { 
        int i = 0; 
        std::cout << i; 
        } 
        Expressions (EXP) - EXP53-CPP. Do not read uninitialized memory 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        98 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
        3.4.3 
        Noncompliant Code Example 
        In this noncompliant code example, an int * object is allocated by a new-expression, but the 
        memory it points to is not initialized. The object’s pointer value and the value it points to are 
        printed to the standard output stream. Printing the pointer value is well-defined, but attempting to 
        print the value pointed to yields an indeterminate value, resulting in undefined behavior. 
        #include <iostream> 
        
        void f() { 
        int *i = new int; 
        std::cout << i << ", " << *i; 
        } 
        3.4.4 
        Compliant Solution 
        In this compliant solution, the memory is direct-initialized to the value 12 prior to printing its 
        value. 
        #include <iostream> 
        
        void f() { 
        int *i = new int(12); 
        std::cout << i << ", " << *i; 
        } 
        Initialization of an object produced by a new-expression is performed by placing (possibly empty) 
        parenthesis or curly braces after the type being allocated. This causes direct initialization of the 
        pointed-to object to occur, which will zero-initialize the object if the initialization omits a value, 
        as illustrated by the following code. 
        int *i = new int(); // zero-initializes *i 
        int *j = new int{}; // zero-initializes *j 
        int *k = new int(12); // initializes *k to 12 
        int *l = new int{12}; // initializes *l to 12 
        Expressions (EXP) - EXP53-CPP. Do not read uninitialized memory 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        99 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
        3.4.5 
        Noncompliant Code Example 
        In this noncompliant code example, the class member variable c is not explicitly initialized by a 
        ctor-initializer in the default constructor. Despite the local variable s being default-initialized, the 
        use of c within the call to S::f() results in the evaluation of an object with indeterminate value, 
        resulting in undefined behavior. 
        class S { 
        int c; 
        
        public: 
        int f(int i) const { return i + c; } 
        }; 
        
        void f() { 
        S s; 
        int i = s.f(10); 
        } 
        3.4.6 
        Compliant Solution 
        In this compliant solution, S is given a default constructor that initializes the class member 
        variable c. 
        class S { 
        int c; 
        
        public: 
        S() : c(0) {} 
        int f(int i) const { return i + c; } 
        }; 
        
        void f() { 
        S s; 
        int i = s.f(10); 
        } 
        3.4.7 
        Risk Assessment 
        Reading uninitialized variables is undefined behavior and can result in unexpected program 
        behavior. In some cases, these security flaws may allow the execution of arbitrary code. 
        Rule 
        Severity 
        Likelihood 
        Remediation Cost 
        Priority 
        Level 
        EXP53-CPP 
        High 
        Probable 
        Medium 
        P12 
        L1 
        Expressions (EXP) - EXP53-CPP. Do not read uninitialized memory 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        100 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
        3.4.8 
        Related Guidelines 
        SEI CERT C Coding Standard 
        EXP33-C. Do not read uninitialized memory 
        3.4.9 
        Bibliography 
        [ISO/IEC 14882-2014] 
        Clause 5, “Expressions” 
        Subclause 5.3.4, “New” 
        Subclause 8.5, “Initializers” 
        Subclause 12.6.2, “Initializing Bases and Members”  
        [Lockheed Martin 2005] 
        Rule 142, All variables shall be initialized before use 
        Expressions (EXP) - EXP54-CPP. Do not access an object outside of its lifetime 
        SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
        
        101 
        Software Engineering Institute | Carnegie Mellon University  
        
        [DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution.
        """,
        "answer": "No",
    },
]

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}"
)

# print(example_prompt.format(**examples[0]))

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

sample_prompt = """Are these two requirements similar? Answer \"Yes\" or \"No\"
DAFMAN Requirement:
8.16.3.  The software shall implement mechanisms to prevent or resolve deadlock conditions 
if the software is responsible for cancelling a Critical Signal after the withdrawal of human 
intent or if the software is responsible for issuing a Safing Command. (T-1) Verification 
activities should prove that the software implements mechanisms such as priority inversion or 
task-assigned resources to prevent or resolve deadlock conditions.
CERT C++ Requirement:
11.1 CON50-CPP. Do not destroy a mutex while it is locked 
Mutex objects are used to protect shared data from being concurrently accessed. If a mutex object 
is destroyed while a thread is blocked waiting for the lock, critical sections and shared data are no 
longer protected. 
The C++ Standard, [thread.mutex.class], paragraph 5 [ISO/IEC 14882-2014], states the following: 
The behavior of a program is undefined if it destroys a mutex object owned by any 
thread or a thread terminates while owning a mutex object. 
Similar wording exists for std::recursive_mutex, std::timed_mutex, 
std::recursive_timed_mutex, and std::shared_timed_mutex. These statements 
imply that destroying a mutex object while a thread is waiting on it is undefined behavior. 
11.1.1 Noncompliant Code Example 
This noncompliant code example creates several threads that each invoke the do_work() 
function, passing a unique number as an ID. 
Unfortunately, this code contains a race condition, allowing the mutex to be destroyed while it is 
still owned, because start_threads() may invoke the mutex’s destructor before all of the 
threads have exited. 
#include <mutex> 
#include <thread> 
 
const size_t maxThreads = 10; 
 
void do_work(size_t i, std::mutex *pm) { 
  std::lock_guard<std::mutex> lk(*pm); 
 
  // Access data protected by the lock. 
} 
 
void start_threads() { 
  std::thread threads[maxThreads]; 
  std::mutex m; 
 
  for (size_t i = 0; i < maxThreads; ++i) { 
    threads[i] = std::thread(do_work, i, &m); 
  } 
} 
Concurrency (CON) - CON50-CPP. Do not destroy a mutex while it is locked 
SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
 
366 
Software Engineering Institute | Carnegie Mellon University  
 
[DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
11.1.2 Compliant Solution 
This compliant solution eliminates the race condition by extending the lifetime of the mutex. 
#include <mutex> 
#include <thread> 
 
const size_t maxThreads = 10; 
 
void do_work(size_t i, std::mutex *pm) { 
  std::lock_guard<std::mutex> lk(*pm); 
 
  // Access data protected by the lock. 
} 
 
std::mutex m; 
 
void start_threads() { 
  std::thread threads[maxThreads]; 
 
  for (size_t i = 0; i < maxThreads; ++i) { 
    threads[i] = std::thread(do_work, i, &m); 
  } 
} 
Concurrency (CON) - CON50-CPP. Do not destroy a mutex while it is locked 
SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
 
367 
Software Engineering Institute | Carnegie Mellon University  
 
[DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
11.1.3 Compliant Solution 
This compliant solution eliminates the race condition by joining the threads before the mutex’s 
destructor is invoked. 
#include <mutex> 
#include <thread> 
 
const size_t maxThreads = 10; 
 
void do_work(size_t i, std::mutex *pm) { 
  std::lock_guard<std::mutex> lk(*pm); 
 
  // Access data protected by the lock. 
} 
void run_threads() { 
  std::thread threads[maxThreads]; 
  std::mutex m; 
 
  for (size_t i = 0; i < maxThreads; ++i) { 
    threads[i] = std::thread(do_work, i, &m); 
  } 
 
  for (size_t i = 0; i < maxThreads; ++i) { 
    threads[i].join(); 
  } 
} 
11.1.4 Risk Assessment 
Destroying a mutex while it is locked may result in invalid control flow and data corruption. 
Rule 
Severity 
Likelihood 
Remediation Cost 
Priority 
Level 
CON50-CPP 
Medium 
Probable 
High 
P4 
L3 
11.1.5 Related Guidelines 
MITRE CWE 
CWE-667, Improper Locking 
SEI CERT C Coding Standard 
CON31-C. Do not destroy a mutex while it is locked 
11.1.6 Bibliography 
[ISO/IEC 14882-2014] 
Subclause 30.4.1, “Mutex Requirements” 
Concurrency (CON) - CON51-CPP. Ensure actively held locks are released on exceptional conditions 
SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
 
368 
Software Engineering Institute | Carnegie Mellon University  
 
[DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution.
Rule Chunk 74:
11.2 CON51-CPP. Ensure actively held locks are released on 
exceptional conditions 
Mutexes that are used to protect accesses to shared data may be locked using the lock() 
member function and unlocked using the unlock() member function. If an exception occurs 
between the call to lock() and the call to unlock(), and the exception changes control flow 
such that unlock() is not called, the mutex will be left in the locked state and no critical 
sections protected by that mutex will be allowed to execute. This is likely to lead to deadlock. 
The throwing of an exception must not allow a mutex to remain locked indefinitely. If a mutex 
was locked and an exception occurs within the critical section protected by that mutex, the mutex 
must be unlocked as part of exception handling before rethrowing the exception or continuing 
execution unless subsequent control flow will unlock the mutex. 
C++ supplies the lock classes lock_guard, unique_lock, and shared_lock, which can 
be initialized with a mutex. In its constructor, the lock object locks the mutex, and in its 
destructor, it unlocks the mutex. The lock_guard class provides a simple RAII wrapper around 
a mutex. The unique_lock and shared_lock classes also use RAII and provide additional 
functionality, such as manual control over the locking strategy. The unique_lock class 
prevents the lock from being copied, although it allows the lock ownership to be moved to another 
lock. The shared_lock class allows the mutex to be shared by several locks. For all three 
classes, if an exception occurs and takes control flow out of the scope of the lock, the destructor 
will unlock the mutex and the program can continue working normally. These lock objects are the 
preferred way to ensure that a mutex is properly released when an exception is thrown. 
11.2.1 Noncompliant Code Example 
This noncompliant code example manipulates shared data and protects the critical section by 
locking the mutex. When it is finished, it unlocks the mutex. However, if an exception occurs 
while manipulating the shared data, the mutex will remain locked. 
#include <mutex> 
 
void manipulate_shared_data(std::mutex &pm) { 
  pm.lock(); 
 
  // Perform work on shared data. 
 
  pm.unlock(); 
} 
Concurrency (CON) - CON51-CPP. Ensure actively held locks are released on exceptional conditions 
SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
 
369 
Software Engineering Institute | Carnegie Mellon University  
 
[DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
11.2.2 Compliant Solution (Manual Unlock) 
This compliant solution catches any exceptions thrown when performing work on the shared data 
and unlocks the mutex before rethrowing the exception. 
#include <mutex> 
 
void manipulate_shared_data(std::mutex &pm) { 
  pm.lock(); 
  try { 
    // Perform work on shared data. 
  } catch (...) { 
    pm.unlock(); 
    throw; 
  } 
  pm.unlock(); // in case no exceptions occur 
} 
11.2.3 Compliant Solution (Lock Object) 
This compliant solution uses a lock_guard object to ensure that the mutex will be unlocked, 
even if an exception occurs, without relying on exception handling machinery and manual 
resource management. 
#include <mutex> 
 
void manipulate_shared_data(std::mutex &pm) { 
  std::lock_guard<std::mutex> lk(pm); 
 
  // Perform work on shared data. 
} 
11.2.4 Risk Assessment 
If an exception occurs while a mutex is locked, deadlock may result. 
Rule 
Severity 
Likelihood 
Remediation Cost 
Priority 
Level 
CON51-CPP 
Low 
Probable 
Low 
P6 
L2 
Concurrency (CON) - CON51-CPP. Ensure actively held locks are released on exceptional conditions 
SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
 
370 
Software Engineering Institute | Carnegie Mellon University  
 
[DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
11.2.5 Related Guidelines 
This rule is a subset of ERR56-CPP. Guarantee exception safety. 
MITRE CWE 
CWE-667, Improper Locking 
SEI CERT Oracle Coding Standard for Java 
LCK08-J. Ensure actively held locks are re-
leased on exceptional conditions 
11.2.6 Bibliography 
[ISO/IEC 14882-2014] 
Subclause 30.4.2, “Locks” 
Concurrency (CON) - CON52-CPP. Prevent data races when accessing bit-fields from multiple threads 
SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
 
371 
Software Engineering Institute | Carnegie Mellon University  
 
[DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution.
Rule Chunk 75:
11.3 CON52-CPP. Prevent data races when accessing bit-fields from 
multiple threads 
When accessing a bit-field, a thread may inadvertently access a separate bit-field in adjacent 
memory. This is because compilers are required to store multiple adjacent bit-fields in one storage 
unit whenever they fit. Consequently, data races may exist not just on a bit-field accessed by 
multiple threads but also on other bit-fields sharing the same byte or word. The problem is 
difficult to diagnose because it may not be obvious that the same memory location is being 
modified by multiple threads. 
One approach for preventing data races in concurrent programming is to use a mutex. When 
properly observed by all threads, a mutex can provide safe and secure access to a shared object. 
However, mutexes provide no guarantees with regard to other objects that might be accessed 
when the mutex is not controlled by the accessing thread. Unfortunately, there is no portable way 
to determine which adjacent bit-fields may be stored along with the desired bit-field. 
Another approach is to insert a non-bit-field member between any two bit-fields to ensure that 
each bit-field is the only one accessed within its storage unit. This technique effectively 
guarantees that no two bit-fields are accessed simultaneously. 
11.3.1 Noncompliant Code Example (bit-field) 
Adjacent bit-fields may be stored in a single memory location. Consequently, modifying adjacent 
bit-fields in different threads is undefined behavior, as shown in this noncompliant code example. 
struct MultiThreadedFlags { 
  unsigned int flag1 : 2; 
  unsigned int flag2 : 2; 
}; 
 
MultiThreadedFlags flags; 
 
void thread1() { 
  flags.flag1 = 1; 
} 
 
void thread2() { 
  flags.flag2 = 2; 
} 
Concurrency (CON) - CON52-CPP. Prevent data races when accessing bit-fields from multiple threads 
SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
 
372 
Software Engineering Institute | Carnegie Mellon University  
 
[DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
For example, the following instruction sequence is possible. 
Thread 1: register 0 = flags 
Thread 1: register 0 &= ~mask(flag1) 
Thread 2: register 0 = flags 
Thread 2: register 0 &= ~mask(flag2) 
Thread 1: register 0 |= 1 << shift(flag1) 
Thread 1: flags = register 0 
Thread 2: register 0 |= 2 << shift(flag2) 
Thread 2: flags = register 0 
11.3.2 Compliant Solution (bit-field, C++11 and later, mutex) 
This compliant solution protects all accesses of the flags with a mutex, thereby preventing any 
data races. 
#include <mutex> 
 
struct MultiThreadedFlags { 
  unsigned int flag1 : 2; 
  unsigned int flag2 : 2; 
}; 
 
struct MtfMutex { 
  MultiThreadedFlags s; 
  std::mutex mutex; 
}; 
 
MtfMutex flags; 
 
void thread1() { 
  std::lock_guard<std::mutex> lk(flags.mutex); 
  flags.s.flag1 = 1; 
} 
  
void thread2() { 
  std::lock_guard<std::mutex> lk(flags.mutex); 
  flags.s.flag2 = 2; 
} 
Concurrency (CON) - CON52-CPP. Prevent data races when accessing bit-fields from multiple threads 
SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
 
373 
Software Engineering Institute | Carnegie Mellon University  
 
[DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
11.3.3 Compliant Solution (C++11) 
In this compliant solution, two threads simultaneously modify two distinct non-bit-field members 
of a structure. Because the members occupy different bytes in memory, no concurrency protection 
is required. 
struct MultiThreadedFlags { 
  unsigned char flag1; 
  unsigned char flag2; 
}; 
  
MultiThreadedFlags flags; 
  
void thread1() { 
  flags.flag1 = 1; 
} 
 
void thread2() { 
  flags.flag2 = 2; 
} 
Unlike earlier versions of the standard, C++11 and later explicitly define a memory location and 
provide the following note in [intro.memory] paragraph 4 [ISO/IEC 14882-2014]: 
[Note: Thus a bit-field and an adjacent non-bit-field are in separate memory locations, 
and therefore can be concurrently updated by two threads of execution without 
interference.  The same applies to two bit-fields, if one is declared inside a nested struct 
declaration and the other is not, or if the two are separated by a zero-length bit-field 
declaration, or if they are separated by a non-bit-field declaration.  It is not safe to 
concurrently update two bit-fields in the same struct if all fields between them are also 
bit-fields of non-zero width.  – end note ] 
It is almost certain that flag1 and flag2 are stored in the same word. Using a compiler that 
conforms to earlier versions of the standard, if both assignments occur on a thread-scheduling 
interleaving that ends with both stores occurring after one another, it is possible that only one of 
the flags will be set as intended, and the other flag will contain its previous value because both 
members are represented by the same word, which is the smallest unit the processor can work on. 
Before the changes made to the C++ Standard for C++11, there were no guarantees that these 
flags could be modified concurrently. 
Concurrency (CON) - CON52-CPP. Prevent data races when accessing bit-fields from multiple threads 
SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
 
374 
Software Engineering Institute | Carnegie Mellon University  
 
[DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution. 
11.3.4 Risk Assessment 
Although the race window is narrow, an assignment or an expression can evaluate improperly 
because of misinterpreted data resulting in a corrupted running state or unintended information 
disclosure. 
Rule 
Severity 
Likelihood 
Remediation Cost 
Priority 
Level 
CON52-CPP 
Medium 
Probable 
Medium 
P8 
L2 
11.3.5 Related Guidelines 
SEI CERT C Coding Standard 
CON32-C. Prevent data races when accessing 
bit-fields from multiple threads 
11.3.6 Bibliography 
[ISO/IEC 14882-2014] 
Subclause 1.7, “The C++ memory model” 
Concurrency (CON) - CON53-CPP. Avoid deadlock by locking in a predefined order 
SEI CERT C++ CODING STANDARD (2016 EDITION) | V01 
 
375 
Software Engineering Institute | Carnegie Mellon University  
 
[DISTRIBUTION STATEMENT A] Approved for public release and unlimited distribution.
"""
print(prompt.format(input=sample_prompt))