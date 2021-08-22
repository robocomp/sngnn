// **********************************************************************
//
// Copyright (c) 2003-2017 ZeroC, Inc. All rights reserved.
//
// This copy of Ice is licensed to you under the terms described in the
// ICE_LICENSE file included in this distribution.
//
// **********************************************************************
//
// Ice version 3.7.0
//
// <auto-generated>
//
// Generated from file `InteractionDetector.ice'
//
// Warning: do not edit this file.
//
// </auto-generated>
//

#ifndef __InteractionDetector_h__
#define __InteractionDetector_h__

#include <IceUtil/PushDisableWarnings.h>
#include <Ice/ProxyF.h>
#include <Ice/ObjectF.h>
#include <Ice/ValueF.h>
#include <Ice/Exception.h>
#include <Ice/LocalObject.h>
#include <Ice/StreamHelpers.h>
#include <Ice/Comparable.h>
#include <Ice/Proxy.h>
#include <Ice/Object.h>
#include <Ice/GCObject.h>
#include <Ice/Value.h>
#include <Ice/Incoming.h>
#include <Ice/FactoryTableInit.h>
#include <IceUtil/ScopedArray.h>
#include <Ice/Optional.h>
#include <IceUtil/UndefSysMacros.h>

#ifndef ICE_IGNORE_VERSION
#   if ICE_INT_VERSION / 100 != 307
#       error Ice version mismatch!
#   endif
#   if ICE_INT_VERSION % 100 > 50
#       error Beta header file detected
#   endif
#   if ICE_INT_VERSION % 100 < 0
#       error Ice patch level mismatch!
#   endif
#endif

#ifdef ICE_CPP11_MAPPING // C++11 mapping

namespace RoboCompInteractionDetector
{

class InteractionDetector;
class InteractionDetectorPrx;

}

namespace RoboCompInteractionDetector
{

struct InteractionT
{
    float timestamp;
    int idSrc;
    ::std::string type;
    int idDst;

    std::tuple<const float&, const int&, const ::std::string&, const int&> ice_tuple() const
    {
        return std::tie(timestamp, idSrc, type, idDst);
    }
};

using InteractionList = ::std::vector<::RoboCompInteractionDetector::InteractionT>;

using Ice::operator<;
using Ice::operator<=;
using Ice::operator>;
using Ice::operator>=;
using Ice::operator==;
using Ice::operator!=;

}

namespace RoboCompInteractionDetector
{

class InteractionDetector : public virtual ::Ice::Object
{
public:

    using ProxyType = InteractionDetectorPrx;

    virtual bool ice_isA(::std::string, const ::Ice::Current&) const override;
    virtual ::std::vector<::std::string> ice_ids(const ::Ice::Current&) const override;
    virtual ::std::string ice_id(const ::Ice::Current&) const override;

    static const ::std::string& ice_staticId();

    virtual void gotinteractions(::RoboCompInteractionDetector::InteractionList, const ::Ice::Current&) = 0;
    bool _iceD_gotinteractions(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual bool _iceDispatch(::IceInternal::Incoming&, const ::Ice::Current&) override;
};

}

namespace RoboCompInteractionDetector
{

class InteractionDetectorPrx : public virtual ::Ice::Proxy<InteractionDetectorPrx, ::Ice::ObjectPrx>
{
public:

    void gotinteractions(const ::RoboCompInteractionDetector::InteractionList& iceP_lst, const ::Ice::Context& context = Ice::noExplicitContext)
    {
        _makePromiseOutgoing<void>(true, this, &RoboCompInteractionDetector::InteractionDetectorPrx::_iceI_gotinteractions, iceP_lst, context).get();
    }

    template<template<typename> class P = ::std::promise>
    auto gotinteractionsAsync(const ::RoboCompInteractionDetector::InteractionList& iceP_lst, const ::Ice::Context& context = Ice::noExplicitContext)
        -> decltype(::std::declval<P<void>>().get_future())
    {
        return _makePromiseOutgoing<void, P>(false, this, &RoboCompInteractionDetector::InteractionDetectorPrx::_iceI_gotinteractions, iceP_lst, context);
    }

    ::std::function<void()>
    gotinteractionsAsync(const ::RoboCompInteractionDetector::InteractionList& iceP_lst,
                         ::std::function<void()> response,
                         ::std::function<void(::std::exception_ptr)> ex = nullptr,
                         ::std::function<void(bool)> sent = nullptr,
                         const ::Ice::Context& context = Ice::noExplicitContext)
    {
        return _makeLamdaOutgoing<void>(response, ex, sent, this, &RoboCompInteractionDetector::InteractionDetectorPrx::_iceI_gotinteractions, iceP_lst, context);
    }

    void _iceI_gotinteractions(const ::std::shared_ptr<::IceInternal::OutgoingAsyncT<void>>&, const ::RoboCompInteractionDetector::InteractionList&, const ::Ice::Context&);

    static const ::std::string& ice_staticId();

protected:

    InteractionDetectorPrx() = default;
    friend ::std::shared_ptr<InteractionDetectorPrx> IceInternal::createProxy<InteractionDetectorPrx>();

    virtual ::std::shared_ptr<::Ice::ObjectPrx> _newInstance() const override;
};

}

namespace Ice
{

template<>
struct StreamableTraits<::RoboCompInteractionDetector::InteractionT>
{
    static const StreamHelperCategory helper = StreamHelperCategoryStruct;
    static const int minWireSize = 13;
    static const bool fixedLength = false;
};

template<typename S>
struct StreamReader<::RoboCompInteractionDetector::InteractionT, S>
{
    static void read(S* istr, ::RoboCompInteractionDetector::InteractionT& v)
    {
        istr->readAll(v.timestamp, v.idSrc, v.type, v.idDst);
    }
};

}

namespace RoboCompInteractionDetector
{

using InteractionDetectorPtr = ::std::shared_ptr<InteractionDetector>;
using InteractionDetectorPrxPtr = ::std::shared_ptr<InteractionDetectorPrx>;

}

#else // C++98 mapping

namespace IceProxy
{

namespace RoboCompInteractionDetector
{

class InteractionDetector;
void _readProxy(::Ice::InputStream*, ::IceInternal::ProxyHandle< ::IceProxy::RoboCompInteractionDetector::InteractionDetector>&);
::IceProxy::Ice::Object* upCast(::IceProxy::RoboCompInteractionDetector::InteractionDetector*);

}

}

namespace RoboCompInteractionDetector
{

class InteractionDetector;
::Ice::Object* upCast(::RoboCompInteractionDetector::InteractionDetector*);
typedef ::IceInternal::Handle< ::RoboCompInteractionDetector::InteractionDetector> InteractionDetectorPtr;
typedef ::IceInternal::ProxyHandle< ::IceProxy::RoboCompInteractionDetector::InteractionDetector> InteractionDetectorPrx;
typedef InteractionDetectorPrx InteractionDetectorPrxPtr;
void _icePatchObjectPtr(InteractionDetectorPtr&, const ::Ice::ObjectPtr&);

}

namespace RoboCompInteractionDetector
{

struct InteractionT
{
    ::Ice::Float timestamp;
    ::Ice::Int idSrc;
    ::std::string type;
    ::Ice::Int idDst;
};

typedef ::std::vector< ::RoboCompInteractionDetector::InteractionT> InteractionList;

}

namespace RoboCompInteractionDetector
{

class Callback_InteractionDetector_gotinteractions_Base : public virtual ::IceInternal::CallbackBase { };
typedef ::IceUtil::Handle< Callback_InteractionDetector_gotinteractions_Base> Callback_InteractionDetector_gotinteractionsPtr;

}

namespace IceProxy
{

namespace RoboCompInteractionDetector
{

class InteractionDetector : public virtual ::Ice::Proxy<InteractionDetector, ::IceProxy::Ice::Object>
{
public:

    void gotinteractions(const ::RoboCompInteractionDetector::InteractionList& iceP_lst, const ::Ice::Context& context = ::Ice::noExplicitContext)
    {
        end_gotinteractions(_iceI_begin_gotinteractions(iceP_lst, context, ::IceInternal::dummyCallback, 0, true));
    }

    ::Ice::AsyncResultPtr begin_gotinteractions(const ::RoboCompInteractionDetector::InteractionList& iceP_lst, const ::Ice::Context& context = ::Ice::noExplicitContext)
    {
        return _iceI_begin_gotinteractions(iceP_lst, context, ::IceInternal::dummyCallback, 0);
    }

    ::Ice::AsyncResultPtr begin_gotinteractions(const ::RoboCompInteractionDetector::InteractionList& iceP_lst, const ::Ice::CallbackPtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_gotinteractions(iceP_lst, ::Ice::noExplicitContext, del, cookie);
    }

    ::Ice::AsyncResultPtr begin_gotinteractions(const ::RoboCompInteractionDetector::InteractionList& iceP_lst, const ::Ice::Context& context, const ::Ice::CallbackPtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_gotinteractions(iceP_lst, context, del, cookie);
    }

    ::Ice::AsyncResultPtr begin_gotinteractions(const ::RoboCompInteractionDetector::InteractionList& iceP_lst, const ::RoboCompInteractionDetector::Callback_InteractionDetector_gotinteractionsPtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_gotinteractions(iceP_lst, ::Ice::noExplicitContext, del, cookie);
    }

    ::Ice::AsyncResultPtr begin_gotinteractions(const ::RoboCompInteractionDetector::InteractionList& iceP_lst, const ::Ice::Context& context, const ::RoboCompInteractionDetector::Callback_InteractionDetector_gotinteractionsPtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_gotinteractions(iceP_lst, context, del, cookie);
    }

    void end_gotinteractions(const ::Ice::AsyncResultPtr&);

private:

    ::Ice::AsyncResultPtr _iceI_begin_gotinteractions(const ::RoboCompInteractionDetector::InteractionList&, const ::Ice::Context&, const ::IceInternal::CallbackBasePtr&, const ::Ice::LocalObjectPtr& cookie = 0, bool sync = false);

public:

    static const ::std::string& ice_staticId();

protected:

    virtual ::IceProxy::Ice::Object* _newInstance() const;
};

}

}

namespace RoboCompInteractionDetector
{

class InteractionDetector : public virtual ::Ice::Object
{
public:

    typedef InteractionDetectorPrx ProxyType;
    typedef InteractionDetectorPtr PointerType;

    virtual ~InteractionDetector();

    virtual bool ice_isA(const ::std::string&, const ::Ice::Current& = ::Ice::emptyCurrent) const;
    virtual ::std::vector< ::std::string> ice_ids(const ::Ice::Current& = ::Ice::emptyCurrent) const;
    virtual const ::std::string& ice_id(const ::Ice::Current& = ::Ice::emptyCurrent) const;

    static const ::std::string& ice_staticId();

    virtual void gotinteractions(const ::RoboCompInteractionDetector::InteractionList&, const ::Ice::Current& = ::Ice::emptyCurrent) = 0;
    bool _iceD_gotinteractions(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual bool _iceDispatch(::IceInternal::Incoming&, const ::Ice::Current&);

protected:

    virtual void _iceWriteImpl(::Ice::OutputStream*) const;
    virtual void _iceReadImpl(::Ice::InputStream*);
};

inline bool operator==(const InteractionDetector& lhs, const InteractionDetector& rhs)
{
    return static_cast<const ::Ice::Object&>(lhs) == static_cast<const ::Ice::Object&>(rhs);
}

inline bool operator<(const InteractionDetector& lhs, const InteractionDetector& rhs)
{
    return static_cast<const ::Ice::Object&>(lhs) < static_cast<const ::Ice::Object&>(rhs);
}

}

namespace Ice
{

template<>
struct StreamableTraits< ::RoboCompInteractionDetector::InteractionT>
{
    static const StreamHelperCategory helper = StreamHelperCategoryStruct;
    static const int minWireSize = 13;
    static const bool fixedLength = false;
};

template<typename S>
struct StreamWriter< ::RoboCompInteractionDetector::InteractionT, S>
{
    static void write(S* ostr, const ::RoboCompInteractionDetector::InteractionT& v)
    {
        ostr->write(v.timestamp);
        ostr->write(v.idSrc);
        ostr->write(v.type);
        ostr->write(v.idDst);
    }
};

template<typename S>
struct StreamReader< ::RoboCompInteractionDetector::InteractionT, S>
{
    static void read(S* istr, ::RoboCompInteractionDetector::InteractionT& v)
    {
        istr->read(v.timestamp);
        istr->read(v.idSrc);
        istr->read(v.type);
        istr->read(v.idDst);
    }
};

}

namespace RoboCompInteractionDetector
{

template<class T>
class CallbackNC_InteractionDetector_gotinteractions : public Callback_InteractionDetector_gotinteractions_Base, public ::IceInternal::OnewayCallbackNC<T>
{
public:

    typedef IceUtil::Handle<T> TPtr;

    typedef void (T::*Exception)(const ::Ice::Exception&);
    typedef void (T::*Sent)(bool);
    typedef void (T::*Response)();

    CallbackNC_InteractionDetector_gotinteractions(const TPtr& obj, Response cb, Exception excb, Sent sentcb)
        : ::IceInternal::OnewayCallbackNC<T>(obj, cb, excb, sentcb)
    {
    }
};

template<class T> Callback_InteractionDetector_gotinteractionsPtr
newCallback_InteractionDetector_gotinteractions(const IceUtil::Handle<T>& instance, void (T::*cb)(), void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_InteractionDetector_gotinteractions<T>(instance, cb, excb, sentcb);
}

template<class T> Callback_InteractionDetector_gotinteractionsPtr
newCallback_InteractionDetector_gotinteractions(const IceUtil::Handle<T>& instance, void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_InteractionDetector_gotinteractions<T>(instance, 0, excb, sentcb);
}

template<class T> Callback_InteractionDetector_gotinteractionsPtr
newCallback_InteractionDetector_gotinteractions(T* instance, void (T::*cb)(), void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_InteractionDetector_gotinteractions<T>(instance, cb, excb, sentcb);
}

template<class T> Callback_InteractionDetector_gotinteractionsPtr
newCallback_InteractionDetector_gotinteractions(T* instance, void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_InteractionDetector_gotinteractions<T>(instance, 0, excb, sentcb);
}

template<class T, typename CT>
class Callback_InteractionDetector_gotinteractions : public Callback_InteractionDetector_gotinteractions_Base, public ::IceInternal::OnewayCallback<T, CT>
{
public:

    typedef IceUtil::Handle<T> TPtr;

    typedef void (T::*Exception)(const ::Ice::Exception& , const CT&);
    typedef void (T::*Sent)(bool , const CT&);
    typedef void (T::*Response)(const CT&);

    Callback_InteractionDetector_gotinteractions(const TPtr& obj, Response cb, Exception excb, Sent sentcb)
        : ::IceInternal::OnewayCallback<T, CT>(obj, cb, excb, sentcb)
    {
    }
};

template<class T, typename CT> Callback_InteractionDetector_gotinteractionsPtr
newCallback_InteractionDetector_gotinteractions(const IceUtil::Handle<T>& instance, void (T::*cb)(const CT&), void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_InteractionDetector_gotinteractions<T, CT>(instance, cb, excb, sentcb);
}

template<class T, typename CT> Callback_InteractionDetector_gotinteractionsPtr
newCallback_InteractionDetector_gotinteractions(const IceUtil::Handle<T>& instance, void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_InteractionDetector_gotinteractions<T, CT>(instance, 0, excb, sentcb);
}

template<class T, typename CT> Callback_InteractionDetector_gotinteractionsPtr
newCallback_InteractionDetector_gotinteractions(T* instance, void (T::*cb)(const CT&), void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_InteractionDetector_gotinteractions<T, CT>(instance, cb, excb, sentcb);
}

template<class T, typename CT> Callback_InteractionDetector_gotinteractionsPtr
newCallback_InteractionDetector_gotinteractions(T* instance, void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_InteractionDetector_gotinteractions<T, CT>(instance, 0, excb, sentcb);
}

}

#endif

#include <IceUtil/PopDisableWarnings.h>
#endif
