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
// Generated from file `PeopleDetector.ice'
//
// Warning: do not edit this file.
//
// </auto-generated>
//

#ifndef __PeopleDetector_h__
#define __PeopleDetector_h__

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

namespace RoboCompPeopleDetector
{

class PeopleDetector;
class PeopleDetectorPrx;

}

namespace RoboCompPeopleDetector
{

struct Person
{
    float timestamp;
    int id;
    float x;
    float y;
    float angle;
    float ix;
    float iy;
    float iangle;

    std::tuple<const float&, const int&, const float&, const float&, const float&, const float&, const float&, const float&> ice_tuple() const
    {
        return std::tie(timestamp, id, x, y, angle, ix, iy, iangle);
    }
};

using PeopleList = ::std::vector<::RoboCompPeopleDetector::Person>;

using Ice::operator<;
using Ice::operator<=;
using Ice::operator>;
using Ice::operator>=;
using Ice::operator==;
using Ice::operator!=;

}

namespace RoboCompPeopleDetector
{

class PeopleDetector : public virtual ::Ice::Object
{
public:

    using ProxyType = PeopleDetectorPrx;

    virtual bool ice_isA(::std::string, const ::Ice::Current&) const override;
    virtual ::std::vector<::std::string> ice_ids(const ::Ice::Current&) const override;
    virtual ::std::string ice_id(const ::Ice::Current&) const override;

    static const ::std::string& ice_staticId();

    virtual void gotpeople(::RoboCompPeopleDetector::PeopleList, const ::Ice::Current&) = 0;
    bool _iceD_gotpeople(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual bool _iceDispatch(::IceInternal::Incoming&, const ::Ice::Current&) override;
};

}

namespace RoboCompPeopleDetector
{

class PeopleDetectorPrx : public virtual ::Ice::Proxy<PeopleDetectorPrx, ::Ice::ObjectPrx>
{
public:

    void gotpeople(const ::RoboCompPeopleDetector::PeopleList& iceP_lst, const ::Ice::Context& context = Ice::noExplicitContext)
    {
        _makePromiseOutgoing<void>(true, this, &RoboCompPeopleDetector::PeopleDetectorPrx::_iceI_gotpeople, iceP_lst, context).get();
    }

    template<template<typename> class P = ::std::promise>
    auto gotpeopleAsync(const ::RoboCompPeopleDetector::PeopleList& iceP_lst, const ::Ice::Context& context = Ice::noExplicitContext)
        -> decltype(::std::declval<P<void>>().get_future())
    {
        return _makePromiseOutgoing<void, P>(false, this, &RoboCompPeopleDetector::PeopleDetectorPrx::_iceI_gotpeople, iceP_lst, context);
    }

    ::std::function<void()>
    gotpeopleAsync(const ::RoboCompPeopleDetector::PeopleList& iceP_lst,
                   ::std::function<void()> response,
                   ::std::function<void(::std::exception_ptr)> ex = nullptr,
                   ::std::function<void(bool)> sent = nullptr,
                   const ::Ice::Context& context = Ice::noExplicitContext)
    {
        return _makeLamdaOutgoing<void>(response, ex, sent, this, &RoboCompPeopleDetector::PeopleDetectorPrx::_iceI_gotpeople, iceP_lst, context);
    }

    void _iceI_gotpeople(const ::std::shared_ptr<::IceInternal::OutgoingAsyncT<void>>&, const ::RoboCompPeopleDetector::PeopleList&, const ::Ice::Context&);

    static const ::std::string& ice_staticId();

protected:

    PeopleDetectorPrx() = default;
    friend ::std::shared_ptr<PeopleDetectorPrx> IceInternal::createProxy<PeopleDetectorPrx>();

    virtual ::std::shared_ptr<::Ice::ObjectPrx> _newInstance() const override;
};

}

namespace Ice
{

template<>
struct StreamableTraits<::RoboCompPeopleDetector::Person>
{
    static const StreamHelperCategory helper = StreamHelperCategoryStruct;
    static const int minWireSize = 32;
    static const bool fixedLength = true;
};

template<typename S>
struct StreamReader<::RoboCompPeopleDetector::Person, S>
{
    static void read(S* istr, ::RoboCompPeopleDetector::Person& v)
    {
        istr->readAll(v.timestamp, v.id, v.x, v.y, v.angle, v.ix, v.iy, v.iangle);
    }
};

}

namespace RoboCompPeopleDetector
{

using PeopleDetectorPtr = ::std::shared_ptr<PeopleDetector>;
using PeopleDetectorPrxPtr = ::std::shared_ptr<PeopleDetectorPrx>;

}

#else // C++98 mapping

namespace IceProxy
{

namespace RoboCompPeopleDetector
{

class PeopleDetector;
void _readProxy(::Ice::InputStream*, ::IceInternal::ProxyHandle< ::IceProxy::RoboCompPeopleDetector::PeopleDetector>&);
::IceProxy::Ice::Object* upCast(::IceProxy::RoboCompPeopleDetector::PeopleDetector*);

}

}

namespace RoboCompPeopleDetector
{

class PeopleDetector;
::Ice::Object* upCast(::RoboCompPeopleDetector::PeopleDetector*);
typedef ::IceInternal::Handle< ::RoboCompPeopleDetector::PeopleDetector> PeopleDetectorPtr;
typedef ::IceInternal::ProxyHandle< ::IceProxy::RoboCompPeopleDetector::PeopleDetector> PeopleDetectorPrx;
typedef PeopleDetectorPrx PeopleDetectorPrxPtr;
void _icePatchObjectPtr(PeopleDetectorPtr&, const ::Ice::ObjectPtr&);

}

namespace RoboCompPeopleDetector
{

struct Person
{
    ::Ice::Float timestamp;
    ::Ice::Int id;
    ::Ice::Float x;
    ::Ice::Float y;
    ::Ice::Float angle;
    ::Ice::Float ix;
    ::Ice::Float iy;
    ::Ice::Float iangle;
};

typedef ::std::vector< ::RoboCompPeopleDetector::Person> PeopleList;

}

namespace RoboCompPeopleDetector
{

class Callback_PeopleDetector_gotpeople_Base : public virtual ::IceInternal::CallbackBase { };
typedef ::IceUtil::Handle< Callback_PeopleDetector_gotpeople_Base> Callback_PeopleDetector_gotpeoplePtr;

}

namespace IceProxy
{

namespace RoboCompPeopleDetector
{

class PeopleDetector : public virtual ::Ice::Proxy<PeopleDetector, ::IceProxy::Ice::Object>
{
public:

    void gotpeople(const ::RoboCompPeopleDetector::PeopleList& iceP_lst, const ::Ice::Context& context = ::Ice::noExplicitContext)
    {
        end_gotpeople(_iceI_begin_gotpeople(iceP_lst, context, ::IceInternal::dummyCallback, 0, true));
    }

    ::Ice::AsyncResultPtr begin_gotpeople(const ::RoboCompPeopleDetector::PeopleList& iceP_lst, const ::Ice::Context& context = ::Ice::noExplicitContext)
    {
        return _iceI_begin_gotpeople(iceP_lst, context, ::IceInternal::dummyCallback, 0);
    }

    ::Ice::AsyncResultPtr begin_gotpeople(const ::RoboCompPeopleDetector::PeopleList& iceP_lst, const ::Ice::CallbackPtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_gotpeople(iceP_lst, ::Ice::noExplicitContext, del, cookie);
    }

    ::Ice::AsyncResultPtr begin_gotpeople(const ::RoboCompPeopleDetector::PeopleList& iceP_lst, const ::Ice::Context& context, const ::Ice::CallbackPtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_gotpeople(iceP_lst, context, del, cookie);
    }

    ::Ice::AsyncResultPtr begin_gotpeople(const ::RoboCompPeopleDetector::PeopleList& iceP_lst, const ::RoboCompPeopleDetector::Callback_PeopleDetector_gotpeoplePtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_gotpeople(iceP_lst, ::Ice::noExplicitContext, del, cookie);
    }

    ::Ice::AsyncResultPtr begin_gotpeople(const ::RoboCompPeopleDetector::PeopleList& iceP_lst, const ::Ice::Context& context, const ::RoboCompPeopleDetector::Callback_PeopleDetector_gotpeoplePtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_gotpeople(iceP_lst, context, del, cookie);
    }

    void end_gotpeople(const ::Ice::AsyncResultPtr&);

private:

    ::Ice::AsyncResultPtr _iceI_begin_gotpeople(const ::RoboCompPeopleDetector::PeopleList&, const ::Ice::Context&, const ::IceInternal::CallbackBasePtr&, const ::Ice::LocalObjectPtr& cookie = 0, bool sync = false);

public:

    static const ::std::string& ice_staticId();

protected:

    virtual ::IceProxy::Ice::Object* _newInstance() const;
};

}

}

namespace RoboCompPeopleDetector
{

class PeopleDetector : public virtual ::Ice::Object
{
public:

    typedef PeopleDetectorPrx ProxyType;
    typedef PeopleDetectorPtr PointerType;

    virtual ~PeopleDetector();

    virtual bool ice_isA(const ::std::string&, const ::Ice::Current& = ::Ice::emptyCurrent) const;
    virtual ::std::vector< ::std::string> ice_ids(const ::Ice::Current& = ::Ice::emptyCurrent) const;
    virtual const ::std::string& ice_id(const ::Ice::Current& = ::Ice::emptyCurrent) const;

    static const ::std::string& ice_staticId();

    virtual void gotpeople(const ::RoboCompPeopleDetector::PeopleList&, const ::Ice::Current& = ::Ice::emptyCurrent) = 0;
    bool _iceD_gotpeople(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual bool _iceDispatch(::IceInternal::Incoming&, const ::Ice::Current&);

protected:

    virtual void _iceWriteImpl(::Ice::OutputStream*) const;
    virtual void _iceReadImpl(::Ice::InputStream*);
};

inline bool operator==(const PeopleDetector& lhs, const PeopleDetector& rhs)
{
    return static_cast<const ::Ice::Object&>(lhs) == static_cast<const ::Ice::Object&>(rhs);
}

inline bool operator<(const PeopleDetector& lhs, const PeopleDetector& rhs)
{
    return static_cast<const ::Ice::Object&>(lhs) < static_cast<const ::Ice::Object&>(rhs);
}

}

namespace Ice
{

template<>
struct StreamableTraits< ::RoboCompPeopleDetector::Person>
{
    static const StreamHelperCategory helper = StreamHelperCategoryStruct;
    static const int minWireSize = 32;
    static const bool fixedLength = true;
};

template<typename S>
struct StreamWriter< ::RoboCompPeopleDetector::Person, S>
{
    static void write(S* ostr, const ::RoboCompPeopleDetector::Person& v)
    {
        ostr->write(v.timestamp);
        ostr->write(v.id);
        ostr->write(v.x);
        ostr->write(v.y);
        ostr->write(v.angle);
        ostr->write(v.ix);
        ostr->write(v.iy);
        ostr->write(v.iangle);
    }
};

template<typename S>
struct StreamReader< ::RoboCompPeopleDetector::Person, S>
{
    static void read(S* istr, ::RoboCompPeopleDetector::Person& v)
    {
        istr->read(v.timestamp);
        istr->read(v.id);
        istr->read(v.x);
        istr->read(v.y);
        istr->read(v.angle);
        istr->read(v.ix);
        istr->read(v.iy);
        istr->read(v.iangle);
    }
};

}

namespace RoboCompPeopleDetector
{

template<class T>
class CallbackNC_PeopleDetector_gotpeople : public Callback_PeopleDetector_gotpeople_Base, public ::IceInternal::OnewayCallbackNC<T>
{
public:

    typedef IceUtil::Handle<T> TPtr;

    typedef void (T::*Exception)(const ::Ice::Exception&);
    typedef void (T::*Sent)(bool);
    typedef void (T::*Response)();

    CallbackNC_PeopleDetector_gotpeople(const TPtr& obj, Response cb, Exception excb, Sent sentcb)
        : ::IceInternal::OnewayCallbackNC<T>(obj, cb, excb, sentcb)
    {
    }
};

template<class T> Callback_PeopleDetector_gotpeoplePtr
newCallback_PeopleDetector_gotpeople(const IceUtil::Handle<T>& instance, void (T::*cb)(), void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_PeopleDetector_gotpeople<T>(instance, cb, excb, sentcb);
}

template<class T> Callback_PeopleDetector_gotpeoplePtr
newCallback_PeopleDetector_gotpeople(const IceUtil::Handle<T>& instance, void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_PeopleDetector_gotpeople<T>(instance, 0, excb, sentcb);
}

template<class T> Callback_PeopleDetector_gotpeoplePtr
newCallback_PeopleDetector_gotpeople(T* instance, void (T::*cb)(), void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_PeopleDetector_gotpeople<T>(instance, cb, excb, sentcb);
}

template<class T> Callback_PeopleDetector_gotpeoplePtr
newCallback_PeopleDetector_gotpeople(T* instance, void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_PeopleDetector_gotpeople<T>(instance, 0, excb, sentcb);
}

template<class T, typename CT>
class Callback_PeopleDetector_gotpeople : public Callback_PeopleDetector_gotpeople_Base, public ::IceInternal::OnewayCallback<T, CT>
{
public:

    typedef IceUtil::Handle<T> TPtr;

    typedef void (T::*Exception)(const ::Ice::Exception& , const CT&);
    typedef void (T::*Sent)(bool , const CT&);
    typedef void (T::*Response)(const CT&);

    Callback_PeopleDetector_gotpeople(const TPtr& obj, Response cb, Exception excb, Sent sentcb)
        : ::IceInternal::OnewayCallback<T, CT>(obj, cb, excb, sentcb)
    {
    }
};

template<class T, typename CT> Callback_PeopleDetector_gotpeoplePtr
newCallback_PeopleDetector_gotpeople(const IceUtil::Handle<T>& instance, void (T::*cb)(const CT&), void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_PeopleDetector_gotpeople<T, CT>(instance, cb, excb, sentcb);
}

template<class T, typename CT> Callback_PeopleDetector_gotpeoplePtr
newCallback_PeopleDetector_gotpeople(const IceUtil::Handle<T>& instance, void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_PeopleDetector_gotpeople<T, CT>(instance, 0, excb, sentcb);
}

template<class T, typename CT> Callback_PeopleDetector_gotpeoplePtr
newCallback_PeopleDetector_gotpeople(T* instance, void (T::*cb)(const CT&), void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_PeopleDetector_gotpeople<T, CT>(instance, cb, excb, sentcb);
}

template<class T, typename CT> Callback_PeopleDetector_gotpeoplePtr
newCallback_PeopleDetector_gotpeople(T* instance, void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_PeopleDetector_gotpeople<T, CT>(instance, 0, excb, sentcb);
}

}

#endif

#include <IceUtil/PopDisableWarnings.h>
#endif
